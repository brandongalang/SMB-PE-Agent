"""
Enhanced semantic search layer for Form ADV data.

This module provides semantic search capabilities over the Form ADV database
using ChromaDB for vector storage and SQLAlchemy for relational queries.
"""

import os
import json
from typing import List, Dict, Any, Optional
import logging

import chromadb
from chromadb.utils import embedding_functions
from sqlalchemy import create_engine, MetaData, Table, select, func, or_
from sqlalchemy.orm import sessionmaker
import sqlalchemy

from adv_models import AdvFirm, DatabaseManager

# Disable ChromaDB anonymous telemetry
os.environ.setdefault("CHROMA_TELEMETRY", "0")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)

class AdvSemanticSearcher:
    """
    Enhanced semantic search layer for Form ADV data.
    
    This class provides semantic search capabilities over the ADV database,
    with support for both vector similarity search and structured queries.
    """
    
    DEFAULT_COLLECTION = "adv_firms"
    
    def __init__(
        self,
        db_url: str = "sqlite:///adv_database.db",
        persist_dir: str = ".chroma_adv_firms",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.db_url = db_url
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize database manager
        self.db_manager = DatabaseManager(db_url)
        self.engine = self.db_manager.engine
        self.SessionLocal = self.db_manager.SessionLocal
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Load existing collection or create empty one
        if self.DEFAULT_COLLECTION in [c.name for c in self.client.list_collections()]:
            self.collection = self.client.get_collection(
                self.DEFAULT_COLLECTION, embedding_function=self.embedding_fn
            )
        else:
            # Create empty collection (don't auto-populate)
            self.collection = self.client.create_collection(
                self.DEFAULT_COLLECTION, embedding_function=self.embedding_fn
            )
    
    def _populate_collection(self, limit: Optional[int] = None) -> None:
        """Populate ChromaDB collection with ADV firm data."""
        logging.info("Populating ChromaDB collection with ADV firm data...")
        
        session = self.SessionLocal()
        try:
            # Get ALL firms with searchable text (complete database rebuild)
            query = session.query(AdvFirm).filter(AdvFirm.searchable_text.isnot(None))
            if limit and limit > 0:
                query = query.limit(limit)
            firms = query.all()
            
            if not firms:
                logging.warning("No firms found with searchable text. Run the processing script first.")
                return
            
            logging.info(f"Found {len(firms)} firms to add to ChromaDB collection.")
            
            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            
            for firm in firms:
                ids.append(str(firm.id))
                documents.append(firm.searchable_text)
                # Create metadata dict, filtering out None values
                metadata = {}
                if firm.filing_id:
                    metadata["filing_id"] = firm.filing_id
                if firm.firm_name:
                    metadata["firm_name"] = firm.firm_name
                if firm.legal_structure:
                    metadata["legal_structure"] = firm.legal_structure
                if firm.address_state:
                    metadata["address_state"] = firm.address_state
                if firm.total_assets is not None:
                    metadata["total_assets"] = float(firm.total_assets)
                if firm.employee_range:
                    metadata["employee_range"] = firm.employee_range
                
                # Add new investment focus fields
                if hasattr(firm, 'focus_sector') and firm.focus_sector:
                    metadata["focus_sector"] = firm.focus_sector
                if hasattr(firm, 'stage_preference') and firm.stage_preference:
                    metadata["stage_preference"] = firm.stage_preference
                if hasattr(firm, 'check_size_range') and firm.check_size_range:
                    metadata["check_size_range"] = firm.check_size_range
                if hasattr(firm, 'target_company_size') and firm.target_company_size:
                    metadata["target_company_size"] = firm.target_company_size
                
                # Add PE-specific fields
                if hasattr(firm, 'is_pe_fund') and firm.is_pe_fund is not None:
                    metadata["is_pe_fund"] = bool(firm.is_pe_fund)
                if hasattr(firm, 'pe_strategy') and firm.pe_strategy:
                    metadata["pe_strategy"] = firm.pe_strategy
                
                # Add check size range as filterable metadata
                min_check, max_check = firm.parse_check_size_range()
                if min_check is not None:
                    metadata["min_check_size_millions"] = float(min_check)
                if max_check is not None:
                    metadata["max_check_size_millions"] = float(max_check)
                
                # Add PE strategy mapping for filtering
                pe_strategy_normalized = firm.get_pe_strategy_mapping()
                if pe_strategy_normalized:
                    metadata["pe_strategy_normalized"] = pe_strategy_normalized
                
                # Add stage compatibility
                compatible_stages = firm.get_stage_alignment()
                if compatible_stages:
                    metadata["compatible_stages"] = compatible_stages
                
                metadatas.append(metadata)
            
            # Add to collection in batches
            BATCH_SIZE = 1000
            total_batches = (len(ids) + BATCH_SIZE - 1) // BATCH_SIZE
            for i in range(0, len(ids), BATCH_SIZE):
                batch_num = i // BATCH_SIZE + 1
                logging.info(f"Processing batch {batch_num}/{total_batches}...")
                
                batch_ids = ids[i:i + BATCH_SIZE]
                batch_docs = documents[i:i + BATCH_SIZE]
                batch_metadata = metadatas[i:i + BATCH_SIZE]
                
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metadata,
                )
            
            logging.info(f"Added {len(ids)} firms to ChromaDB collection.")
            
        except Exception as e:
            logging.error(f"Error populating collection: {e}")
            raise
        finally:
            session.close()
    
    def rebuild_collection(self, limit: Optional[int] = None) -> None:
        """Rebuild the ChromaDB collection from scratch."""
        logging.info("Rebuilding ChromaDB collection...")
        
        # Delete existing collection
        try:
            self.client.delete_collection(self.DEFAULT_COLLECTION)
        except Exception:
            pass  # Collection might not exist
        
        # Create new collection
        self.collection = self.client.create_collection(
            self.DEFAULT_COLLECTION, embedding_function=self.embedding_fn
        )
        
        # Populate with current data
        self._populate_collection(limit=limit)
    
    def semantic_search(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search over ADV firms.
        
        Args:
            query: Search query string
            n_results: Number of results to return
            filters: Optional metadata filters (e.g., {"legal_structure": "Corporation"})
            include_metadata: Whether to include full firm data
            
        Returns:
            List of search results with scores and firm data
        """
        # Perform vector search
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filters,
            include=["distances", "metadatas"]
        )
        
        # Extract results
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        
        output = []
        
        if include_metadata and ids:
            # Get full firm data from database
            session = self.SessionLocal()
            try:
                firm_ids = [int(id_) for id_ in ids]
                firms = session.query(AdvFirm).filter(AdvFirm.id.in_(firm_ids)).all()
                firms_by_id = {firm.id: firm for firm in firms}
                
                for i, firm_id in enumerate(firm_ids):
                    firm = firms_by_id.get(firm_id)
                    if firm:
                        result = {
                            "score": distances[i],
                            "firm": firm.to_dict(),
                            "searchable_text": firm.searchable_text,
                            "metadata": metadatas[i] if metadatas else {}
                        }
                        output.append(result)
                        
            except Exception as e:
                logging.error(f"Error retrieving firm data: {e}")
                # Fallback to metadata only
                for i, firm_id in enumerate(ids):
                    result = {
                        "score": distances[i],
                        "firm_id": firm_id,
                        "metadata": metadatas[i] if metadatas else {}
                    }
                    output.append(result)
            finally:
                session.close()
        else:
            # Return basic results without full firm data
            for i, firm_id in enumerate(ids):
                result = {
                    "score": distances[i],
                    "firm_id": firm_id,
                    "metadata": metadatas[i] if metadatas else {}
                }
                output.append(result)
        
        return output
    
    def search_pe_funds(
        self,
        query: str,
        pe_strategy: Optional[str] = None,
        stage_preference: Optional[str] = None,
        min_check_size: Optional[float] = None,
        max_check_size: Optional[float] = None,
        target_company_size: Optional[str] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search specifically for PE funds with PE-specific criteria.
        
        Args:
            query: Semantic search query
            pe_strategy: PE strategy (e.g., "Buyout", "Growth Equity", "Venture Capital")
            stage_preference: Investment stage preference
            min_check_size: Minimum check size in millions USD
            max_check_size: Maximum check size in millions USD
            target_company_size: Target company size
            n_results: Number of results to return
            
        Returns:
            List of PE fund matches with scores and metadata
        """
        # Build ChromaDB filters for PE-specific search
        # Use $and operator for multiple conditions to avoid ChromaDB query errors
        filter_conditions = [{"is_pe_fund": True}]  # Only return PE funds
        
        # Add additional filters as separate conditions
        if pe_strategy:
            filter_conditions.append({"pe_strategy_normalized": pe_strategy})
        
        if target_company_size:
            filter_conditions.append({"target_company_size": target_company_size})
        
        # Construct proper ChromaDB filter syntax
        if len(filter_conditions) == 1:
            filters = filter_conditions[0]
        else:
            filters = {"$and": filter_conditions}
        
        # Note: stage_preference, min/max_check_size will be handled in post-processing
        # due to ChromaDB's limited support for complex filter operators
        
        # Perform semantic search with progressive filter relaxation
        # Get more results initially to allow for post-filtering
        search_results = self._search_with_progressive_relaxation(
            query=query,
            base_filters=filters,
            n_results=n_results * 3,  # Get 3x to allow for filtering
            pe_strategy=pe_strategy,
            stage_preference=stage_preference,
            min_check_size=min_check_size,
            max_check_size=max_check_size
        )
        
        # Apply post-processing filters for criteria not handled by ChromaDB
        enhanced_results = []
        for result in search_results:
            firm = result.get("firm", {})
            
            # Apply stage preference filter
            if stage_preference:
                compatible_stages = firm.get("compatible_stages", "").split(",")
                compatible_stages = [s.strip() for s in compatible_stages if s.strip()]
                if stage_preference not in compatible_stages:
                    continue
            
            # Apply check size filters
            if min_check_size is not None:
                firm_max_check = firm.get("max_check_size_millions")
                if firm_max_check is None or firm_max_check < min_check_size:
                    continue
            
            if max_check_size is not None:
                firm_min_check = firm.get("min_check_size_millions")
                if firm_min_check is None or firm_min_check > max_check_size:
                    continue
            
            # Add PE-specific analysis to each result
            enhanced_result = result.copy()
            enhanced_result["pe_analysis"] = self._analyze_pe_fit(
                firm, query, pe_strategy, stage_preference, min_check_size, max_check_size
            )
            
            enhanced_results.append(enhanced_result)
            
            # Stop when we have enough results
            if len(enhanced_results) >= n_results:
                break
        
        return enhanced_results[:n_results]
    
    def _search_with_progressive_relaxation(
        self,
        query: str,
        base_filters: Dict[str, Any],
        n_results: int,
        pe_strategy: Optional[str] = None,
        stage_preference: Optional[str] = None,
        min_check_size: Optional[float] = None,
        max_check_size: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform search with progressive filter relaxation to ensure results.
        
        Strategy:
        1. Try with all filters
        2. Remove size constraints if no results
        3. Remove stage preference if still no results
        4. Remove strategy constraint if still no results
        5. Fall back to PE funds only if still no results
        """
        # Level 1: Try with all filters
        results = self.semantic_search(
            query=query,
            n_results=n_results,
            filters=base_filters,
            include_metadata=True
        )
        
        if results:
            return results
        
        # Level 2: Remove target company size constraint
        # Rebuild filters without target_company_size
        filter_conditions = [{"is_pe_fund": True}]
        if pe_strategy:
            filter_conditions.append({"pe_strategy_normalized": pe_strategy})
        
        relaxed_filters = filter_conditions[0] if len(filter_conditions) == 1 else {"$and": filter_conditions}
        
        results = self.semantic_search(
            query=query,
            n_results=n_results,
            filters=relaxed_filters,
            include_metadata=True
        )
        
        if results:
            return results
        
        # Level 3: Remove PE strategy constraint (PE funds only)
        minimal_filters = {"is_pe_fund": True}
        
        results = self.semantic_search(
            query=query,
            n_results=n_results,
            filters=minimal_filters,
            include_metadata=True
        )
        
        if results:
            return results
        
        # Level 4: PE funds only (fallback)
        fallback_filters = {"is_pe_fund": True}
        
        results = self.semantic_search(
            query=query,
            n_results=n_results,
            filters=fallback_filters,
            include_metadata=True
        )
        
        return results if results else []
    
    def search_comprehensive(
        self,
        query: str,
        pe_criteria: Optional[Dict[str, Any]] = None,
        firm_criteria: Optional[Dict[str, Any]] = None,
        geographic_criteria: Optional[Dict[str, Any]] = None,
        n_results: int = 10,
        require_pe_funds: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Comprehensive multi-criteria search combining semantic, PE-specific, and firm filters.
        
        Args:
            query: Semantic search query
            pe_criteria: PE-specific filters {
                'strategy': str, 'stage_preference': str, 
                'min_check_size': float, 'max_check_size': float,
                'target_company_size': str
            }
            firm_criteria: Firm-level filters {
                'min_assets': float, 'max_assets': float,
                'legal_structures': List[str], 'employee_range': str
            }
            geographic_criteria: Geographic filters {
                'states': List[str], 'cities': List[str]
            }
            n_results: Number of results to return
            require_pe_funds: Whether to restrict to PE funds only
            
        Returns:
            List of comprehensive matches with multi-dimensional scoring
        """
        # Build comprehensive filter set
        base_filters = {}
        
        # PE fund requirement
        if require_pe_funds:
            base_filters["is_pe_fund"] = True
        
        # PE-specific criteria
        if pe_criteria:
            if pe_criteria.get('strategy'):
                base_filters["pe_strategy_normalized"] = pe_criteria['strategy']
            if pe_criteria.get('stage_preference'):
                base_filters["compatible_stages"] = {"$contains": pe_criteria['stage_preference']}
            if pe_criteria.get('min_check_size') is not None:
                base_filters["max_check_size_millions"] = {"$gte": pe_criteria['min_check_size']}
            if pe_criteria.get('max_check_size') is not None:
                base_filters["min_check_size_millions"] = {"$lte": pe_criteria['max_check_size']}
            if pe_criteria.get('target_company_size'):
                base_filters["target_company_size"] = pe_criteria['target_company_size']
        
        # Geographic criteria
        if geographic_criteria:
            if geographic_criteria.get('states'):
                # Create OR filter for multiple states
                states = geographic_criteria['states']
                if len(states) == 1:
                    base_filters["address_state"] = states[0]
                # Note: ChromaDB doesn't support complex OR queries in metadata, 
                # so we'll handle multiple states in post-processing
        
        # Firm-level criteria will be handled via database query + semantic search
        # For now, use the enhanced search method
        
        if pe_criteria:
            # Use PE-specific search for optimal filtering
            results = self.search_pe_funds(
                query=query,
                pe_strategy=pe_criteria.get('strategy'),
                stage_preference=pe_criteria.get('stage_preference'),
                min_check_size=pe_criteria.get('min_check_size'),
                max_check_size=pe_criteria.get('max_check_size'),
                target_company_size=pe_criteria.get('target_company_size'),
                n_results=n_results * 2  # Get more for post-filtering
            )
        else:
            # Use basic semantic search
            results = self.semantic_search(
                query=query,
                n_results=n_results * 2,
                filters=base_filters,
                include_metadata=True
            )
        
        # Post-process with firm and geographic criteria
        filtered_results = self._apply_comprehensive_filters(
            results, firm_criteria, geographic_criteria
        )
        
        # Add comprehensive scoring
        scored_results = []
        for result in filtered_results[:n_results]:
            comprehensive_result = result.copy()
            comprehensive_result["comprehensive_score"] = self._calculate_comprehensive_score(
                result, pe_criteria, firm_criteria, geographic_criteria
            )
            comprehensive_result["match_breakdown"] = self._generate_match_breakdown(
                result, pe_criteria, firm_criteria, geographic_criteria
            )
            scored_results.append(comprehensive_result)
        
        # Sort by comprehensive score
        scored_results.sort(key=lambda x: x.get("comprehensive_score", 0), reverse=True)
        
        return scored_results
    
    def _apply_comprehensive_filters(
        self,
        results: List[Dict[str, Any]],
        firm_criteria: Optional[Dict[str, Any]],
        geographic_criteria: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply post-search filtering for criteria not handled by ChromaDB."""
        filtered = []
        
        for result in results:
            firm = result.get("firm", {})
            
            # Apply firm criteria
            if firm_criteria:
                if firm_criteria.get('min_assets') and firm.get('total_assets'):
                    if firm['total_assets'] < firm_criteria['min_assets']:
                        continue
                
                if firm_criteria.get('max_assets') and firm.get('total_assets'):
                    if firm['total_assets'] > firm_criteria['max_assets']:
                        continue
                
                if firm_criteria.get('legal_structures'):
                    if firm.get('legal_structure') not in firm_criteria['legal_structures']:
                        continue
                
                if firm_criteria.get('employee_range'):
                    if firm.get('employee_range') != firm_criteria['employee_range']:
                        continue
            
            # Apply geographic criteria
            if geographic_criteria:
                if geographic_criteria.get('states'):
                    firm_state = firm.get('address_state')
                    if firm_state not in geographic_criteria['states']:
                        continue
                
                if geographic_criteria.get('cities'):
                    firm_city = firm.get('address_city')
                    if firm_city not in geographic_criteria['cities']:
                        continue
            
            filtered.append(result)
        
        return filtered
    
    def _calculate_comprehensive_score(
        self,
        result: Dict[str, Any],
        pe_criteria: Optional[Dict[str, Any]],
        firm_criteria: Optional[Dict[str, Any]],
        geographic_criteria: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate comprehensive match score across all criteria."""
        base_score = 1.0 - result.get("score", 1.0)  # Convert distance to similarity
        
        # PE criteria bonus (up to +0.5)
        pe_bonus = 0.0
        if result.get("pe_analysis"):
            pe_analysis = result["pe_analysis"]
            if pe_analysis.get("strategy_match"):
                pe_bonus += 0.2
            if pe_analysis.get("stage_match"):
                pe_bonus += 0.15
            if pe_analysis.get("size_match"):
                pe_bonus += 0.15
        
        # Firm criteria bonus (up to +0.3)
        firm_bonus = 0.0
        if firm_criteria:
            firm = result.get("firm", {})
            
            # Asset size alignment
            if firm_criteria.get('min_assets') and firm.get('total_assets'):
                if firm['total_assets'] >= firm_criteria['min_assets']:
                    firm_bonus += 0.1
            
            # Legal structure preference
            if firm_criteria.get('legal_structures') and firm.get('legal_structure'):
                if firm['legal_structure'] in firm_criteria['legal_structures']:
                    firm_bonus += 0.1
            
            # Employee size preference
            if firm_criteria.get('employee_range') and firm.get('employee_range'):
                if firm['employee_range'] == firm_criteria['employee_range']:
                    firm_bonus += 0.1
        
        # Geographic bonus (up to +0.2)
        geo_bonus = 0.0
        if geographic_criteria:
            firm = result.get("firm", {})
            
            if geographic_criteria.get('states') and firm.get('address_state'):
                if firm['address_state'] in geographic_criteria['states']:
                    geo_bonus += 0.15
            
            if geographic_criteria.get('cities') and firm.get('address_city'):
                if firm['address_city'] in geographic_criteria['cities']:
                    geo_bonus += 0.05
        
        return min(1.0, base_score + pe_bonus + firm_bonus + geo_bonus)
    
    def _generate_match_breakdown(
        self,
        result: Dict[str, Any],
        pe_criteria: Optional[Dict[str, Any]],
        firm_criteria: Optional[Dict[str, Any]],
        geographic_criteria: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate detailed match breakdown explanation."""
        breakdown = {
            "semantic_match": 1.0 - result.get("score", 1.0),
            "pe_match_details": {},
            "firm_match_details": {},
            "geographic_match_details": {},
            "overall_reasoning": []
        }
        
        # PE match details
        if result.get("pe_analysis"):
            breakdown["pe_match_details"] = result["pe_analysis"]
            if result["pe_analysis"].get("reasoning"):
                breakdown["overall_reasoning"].extend(result["pe_analysis"]["reasoning"])
        
        # Firm match details
        firm = result.get("firm", {})
        if firm_criteria:
            if firm.get('total_assets') and firm_criteria.get('min_assets'):
                breakdown["firm_match_details"]["assets_sufficient"] = firm['total_assets'] >= firm_criteria['min_assets']
            if firm.get('legal_structure') and firm_criteria.get('legal_structures'):
                breakdown["firm_match_details"]["legal_structure_match"] = firm['legal_structure'] in firm_criteria['legal_structures']
        
        # Geographic match details
        if geographic_criteria:
            if firm.get('address_state') and geographic_criteria.get('states'):
                breakdown["geographic_match_details"]["state_match"] = firm['address_state'] in geographic_criteria['states']
            if firm.get('address_city') and geographic_criteria.get('cities'):
                breakdown["geographic_match_details"]["city_match"] = firm['address_city'] in geographic_criteria['cities']
        
        return breakdown
    
    def _analyze_pe_fit(
        self,
        firm: Dict[str, Any],
        query: str,
        pe_strategy: Optional[str] = None,
        stage_preference: Optional[str] = None,
        min_check_size: Optional[float] = None,
        max_check_size: Optional[float] = None
    ) -> Dict[str, Any]:
        """Analyze PE fund fit with specific criteria."""
        analysis = {
            "strategy_match": False,
            "stage_match": False,
            "size_match": False,
            "fit_score": 0.0,
            "reasoning": []
        }
        
        # Strategy alignment
        if pe_strategy and firm.get("pe_strategy_normalized") == pe_strategy:
            analysis["strategy_match"] = True
            analysis["fit_score"] += 0.4
            analysis["reasoning"].append(f"Perfect strategy match: {pe_strategy}")
        elif firm.get("pe_strategy_normalized"):
            analysis["reasoning"].append(f"Strategy: {firm.get('pe_strategy_normalized')}")
        
        # Stage alignment
        compatible_stages = firm.get("compatible_stages", "").split(",")
        if stage_preference and stage_preference in compatible_stages:
            analysis["stage_match"] = True
            analysis["fit_score"] += 0.3
            analysis["reasoning"].append(f"Stage compatible: {stage_preference}")
        elif compatible_stages:
            analysis["reasoning"].append(f"Compatible stages: {', '.join(compatible_stages)}")
        
        # Size alignment
        firm_min = firm.get("min_check_size_millions")
        firm_max = firm.get("max_check_size_millions")
        
        size_compatible = True
        if min_check_size and firm_max and min_check_size > firm_max:
            size_compatible = False
        if max_check_size and firm_min and max_check_size < firm_min:
            size_compatible = False
        
        if size_compatible:
            analysis["size_match"] = True
            analysis["fit_score"] += 0.3
            if firm_min and firm_max:
                analysis["reasoning"].append(f"Check size range: ${firm_min:.1f}M-${firm_max:.1f}M")
        else:
            if firm_min and firm_max:
                analysis["reasoning"].append(f"Size mismatch: Fund checks ${firm_min:.1f}M-${firm_max:.1f}M")
        
        return analysis
    
    def search_by_criteria(
        self,
        query: str,
        asset_range: Optional[tuple] = None,
        legal_structures: Optional[List[str]] = None,
        states: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search ADV firms with both semantic and structured criteria.
        
        Args:
            query: Semantic search query
            asset_range: Tuple of (min_assets, max_assets) in USD
            legal_structures: List of legal structures to filter by
            states: List of states to filter by
            strategies: List of investment strategies to filter by
            n_results: Number of results to return
            
        Returns:
            List of matching firms with scores and metadata
        """
        session = self.SessionLocal()
        try:
            # Build base query
            base_query = session.query(AdvFirm)
            
            # Apply structured filters
            if asset_range:
                min_assets, max_assets = asset_range
                if min_assets is not None:
                    base_query = base_query.filter(AdvFirm.total_assets >= min_assets)
                if max_assets is not None:
                    base_query = base_query.filter(AdvFirm.total_assets <= max_assets)
            
            if legal_structures:
                base_query = base_query.filter(AdvFirm.legal_structure.in_(legal_structures))
            
            if states:
                # Filter by state registration or headquarters
                state_filter = []
                for state in states:
                    state_filter.append(AdvFirm.address_state == state)
                    state_filter.append(AdvFirm.registered_states.contains(state))
                base_query = base_query.filter(
                    or_(*state_filter) if state_filter else True
                )
            
            # Get candidate firms
            candidate_firms = base_query.all()
            
            if not candidate_firms:
                return []
            
            # Perform semantic search on candidates
            candidate_ids = [str(firm.id) for firm in candidate_firms]
            
            # Use ChromaDB to search within candidates
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, len(candidate_ids)),
                where={"$and": [{"id": {"$in": candidate_ids}}]} if candidate_ids else None,
                include=["distances", "metadatas"]
            )
            
            # Process results
            output = []
            if results["ids"]:
                ids = results["ids"][0]
                distances = results["distances"][0]
                
                firms_by_id = {firm.id: firm for firm in candidate_firms}
                
                for i, firm_id in enumerate(ids):
                    firm = firms_by_id.get(int(firm_id))
                    if firm:
                        result = {
                            "score": distances[i],
                            "firm": firm.to_dict(),
                            "searchable_text": firm.searchable_text,
                            "matches_criteria": {
                                "asset_range": asset_range,
                                "legal_structures": legal_structures,
                                "states": states,
                                "strategies": strategies
                            }
                        }
                        output.append(result)
            
            return output
            
        except Exception as e:
            logging.error(f"Error in criteria search: {e}")
            return []
        finally:
            session.close()
    
    def get_firm_by_id(self, firm_id: int) -> Optional[AdvFirm]:
        """Get a specific firm by ID."""
        session = self.SessionLocal()
        try:
            return session.query(AdvFirm).filter(AdvFirm.id == firm_id).first()
        finally:
            session.close()
    
    def get_firm_by_filing_id(self, filing_id: str) -> Optional[AdvFirm]:
        """Get a specific firm by filing ID."""
        session = self.SessionLocal()
        try:
            return session.query(AdvFirm).filter(AdvFirm.filing_id == filing_id).first()
        finally:
            session.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        session = self.SessionLocal()
        try:
            total_firms = session.query(AdvFirm).count()
            with_searchable_text = session.query(AdvFirm).filter(
                AdvFirm.searchable_text.isnot(None)
            ).count()
            
            # Legal structure distribution
            legal_structures = session.query(
                AdvFirm.legal_structure,
                func.count(AdvFirm.legal_structure)
            ).group_by(AdvFirm.legal_structure).all()
            
            # State distribution
            states = session.query(
                AdvFirm.address_state,
                func.count(AdvFirm.address_state)
            ).group_by(AdvFirm.address_state).limit(10).all()
            
            return {
                "total_firms": total_firms,
                "firms_with_searchable_text": with_searchable_text,
                "chromadb_collection_count": self.collection.count(),
                "legal_structures": dict(legal_structures),
                "top_states": dict(states)
            }
        finally:
            session.close()


def main():
    """Command-line interface for ADV semantic search."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Search ADV firms semantically")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--k", type=int, default=10, help="Number of results")
    parser.add_argument("--db-url", default="sqlite:///adv_database.db", help="Database URL")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild ChromaDB collection")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--legal-structure", help="Filter by legal structure")
    parser.add_argument("--state", help="Filter by state")
    parser.add_argument("--min-assets", type=float, help="Minimum assets under management")
    parser.add_argument("--max-assets", type=float, help="Maximum assets under management")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to process")
    
    args = parser.parse_args()
    
    # Initialize searcher
    searcher = AdvSemanticSearcher(db_url=args.db_url)
    
    if args.rebuild:
        searcher.rebuild_collection(limit=args.limit)
        print("ChromaDB collection rebuilt successfully.")
        return
    
    if args.stats:
        stats = searcher.get_stats()
        print(json.dumps(stats, indent=2))
        return
    
    # Build filters
    filters = {}
    if args.legal_structure:
        filters["legal_structure"] = args.legal_structure
    if args.state:
        filters["address_state"] = args.state
    
    # Perform search
    if args.min_assets or args.max_assets:
        # Use criteria search
        results = searcher.search_by_criteria(
            query=args.query,
            asset_range=(args.min_assets, args.max_assets),
            legal_structures=[args.legal_structure] if args.legal_structure else None,
            states=[args.state] if args.state else None,
            n_results=args.k
        )
    else:
        # Use basic semantic search
        results = searcher.semantic_search(
            query=args.query,
            n_results=args.k,
            filters=filters if filters else None
        )
    
    # Display results
    print(f"Found {len(results)} results for query: '{args.query}'")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        firm = result.get("firm", {})
        print(f"{i}. {firm.get('firm_name', 'N/A')} (Score: {result['score']:.4f})")
        print(f"   Legal Structure: {firm.get('legal_structure', 'N/A')}")
        print(f"   Location: {firm.get('address_city', 'N/A')}, {firm.get('address_state', 'N/A')}")
        assets = firm.get('total_assets')
        if assets is not None:
            print(f"   Assets: ${assets:,.0f}")
        else:
            print(f"   Assets: N/A")
        print(f"   Employees: {firm.get('employee_range', 'N/A')}")
        if result.get("searchable_text"):
            print(f"   Context: {result['searchable_text'][:200]}...")
        print()


if __name__ == "__main__":
    main()