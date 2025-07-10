"""
Enhanced fund matcher that works with both synthetic PE funds and real ADV data.

This module provides an enhanced matching system that can leverage both the 
existing synthetic PE fund database and the real Form ADV data for better
company-to-fund matching capabilities.
"""

import os
from typing import List, Dict, Any, Optional, Union
import logging

# Disable Chroma telemetry
os.environ.setdefault("CHROMA_TELEMETRY", "0")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)

# from data_models import ProcessedCompanyData  # DEPRECATED - using SMBCompanyData instead
from api.agents.smb_agent import SMBCompanyData
from pe_fund_matcher import FundMatcher, SemanticFundSearcher
from adv_semantic_layer import AdvSemanticSearcher

class EnhancedFundMatcher:
    """
    Enhanced fund matcher that combines synthetic PE funds with real ADV data.
    
    This matcher can search both:
    1. Synthetic PE fund database (existing functionality)
    2. Real Form ADV data from SEC filings
    
    It provides ranking, scoring, and reasoning for matches from both sources.
    """
    
    def __init__(
        self,
        synthetic_db_path: str = "sec_adv_synthetic.db",
        adv_db_url: str = "sqlite:///adv_database.db",
        use_synthetic: bool = True,
        use_adv: bool = True
    ):
        """
        Initialize the enhanced fund matcher.
        
        Args:
            synthetic_db_path: Path to synthetic PE fund database
            adv_db_url: URL to ADV database
            use_synthetic: Whether to include synthetic PE fund matches
            use_adv: Whether to include ADV firm matches
        """
        self.use_synthetic = use_synthetic
        self.use_adv = use_adv
        
        # Initialize synthetic fund matcher if enabled
        if self.use_synthetic:
            try:
                self.synthetic_matcher = FundMatcher()
                logging.info("Synthetic fund matcher initialized successfully.")
            except Exception as e:
                logging.warning(f"Failed to initialize synthetic fund matcher: {e}")
                self.use_synthetic = False
                self.synthetic_matcher = None
        else:
            self.synthetic_matcher = None
        
        # Initialize ADV searcher if enabled
        if self.use_adv:
            try:
                self.adv_searcher = AdvSemanticSearcher(db_url=adv_db_url)
                logging.info("ADV semantic searcher initialized successfully.")
            except Exception as e:
                logging.warning(f"Failed to initialize ADV searcher: {e}")
                self.use_adv = False
                self.adv_searcher = None
        else:
            self.adv_searcher = None
    
    def match_company(
        self,
        company: SMBCompanyData,
        k_synthetic: int = 5,
        k_adv: int = 10,
        total_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Find matching funds/firms for a company using both data sources.
        
        Args:
            company: Company data to match
            k_synthetic: Number of synthetic fund matches to return
            k_adv: Number of ADV firm matches to return  
            total_k: If specified, limit total results across both sources
            
        Returns:
            Dictionary with synthetic_matches, adv_matches, and combined ranking
        """
        results = {
            "company": company.model_dump() if hasattr(company, 'model_dump') else str(company),
            "synthetic_matches": [],
            "adv_matches": [],
            "combined_ranking": [],
            "match_summary": {}
        }
        
        # Get synthetic PE fund matches
        if self.use_synthetic and self.synthetic_matcher:
            try:
                synthetic_matches = self.synthetic_matcher.match(company, k=k_synthetic)
                results["synthetic_matches"] = self._enhance_synthetic_matches(
                    synthetic_matches, company
                )
                logging.info(f"Found {len(synthetic_matches)} synthetic fund matches.")
            except Exception as e:
                logging.error(f"Error getting synthetic matches: {e}")
                results["synthetic_matches"] = []
        
        # Get ADV firm matches
        if self.use_adv and self.adv_searcher:
            try:
                adv_matches = self._get_adv_matches(company, k_adv)
                results["adv_matches"] = adv_matches
                logging.info(f"Found {len(adv_matches)} ADV firm matches.")
            except Exception as e:
                logging.error(f"Error getting ADV matches: {e}")
                results["adv_matches"] = []
        
        # Create combined ranking
        results["combined_ranking"] = self._create_combined_ranking(
            results["synthetic_matches"], 
            results["adv_matches"],
            total_k
        )
        
        # Generate match summary
        results["match_summary"] = self._generate_match_summary(
            results["synthetic_matches"],
            results["adv_matches"], 
            company
        )
        
        return results
    
    def _get_adv_matches(self, company: SMBCompanyData, k: int) -> List[Dict[str, Any]]:
        """Get ADV firm matches for a company using enhanced PE-specific search."""
        # Build search query from company data
        query_parts = []
        
        # Add industry information
        if company.primary_industry:
            query_parts.append(company.primary_industry)
        
        query_parts.extend(company.industry_keywords[:3])
        
        # Add business model and stage
        if company.business_model and company.business_model != "Unknown":
            query_parts.append(company.business_model)
        
        if company.growth_stage and company.growth_stage != "Unknown":
            query_parts.append(company.growth_stage)
        
        query = " ".join(filter(None, query_parts))
        
        # Map company characteristics to PE criteria
        pe_strategy = self._map_company_to_pe_strategy(company)
        stage_preference = self._map_company_stage_to_pe_stage(company.growth_stage)
        min_check_size, max_check_size = self._estimate_check_size_range(company)
        target_company_size = self._map_company_size_to_target(company)
        
        # Use PE-specific search with enhanced filtering
        if hasattr(self.adv_searcher, 'search_pe_funds'):
            matches = self.adv_searcher.search_pe_funds(
                query=query,
                pe_strategy=pe_strategy,
                stage_preference=stage_preference,
                min_check_size=min_check_size,
                max_check_size=max_check_size,
                target_company_size=target_company_size,
                n_results=k
            )
        else:
            # Fallback to basic search with PE filters
            filters = {"is_pe_fund": True}
            matches = self.adv_searcher.semantic_search(
                query=query,
                n_results=k,
                filters=filters,
                include_metadata=True
            )
        
        # Enhance matches with reasoning
        enhanced_matches = []
        for match in matches:
            enhanced_match = self._enhance_adv_match(match, company)
            enhanced_matches.append(enhanced_match)
        
        return enhanced_matches
    
    def _map_company_to_pe_strategy(self, company: SMBCompanyData) -> Optional[str]:
        """Map company characteristics to likely PE strategy."""
        # Map based on company stage and characteristics
        stage = company.growth_stage.lower() if company.growth_stage else ""
        
        if any(word in stage for word in ["seed", "startup", "early"]):
            return "Venture Capital"
        elif any(word in stage for word in ["growth", "expansion", "scaling"]):
            return "Growth Equity"
        elif any(word in stage for word in ["mature", "established", "profitable"]):
            return "Buyout"
        elif any(word in stage for word in ["distressed", "turnaround"]):
            return "Distressed"
        
        # Default based on company size
        if company.estimated_revenue:
            if company.estimated_revenue < 10000000:  # < $10M
                return "Venture Capital"
            elif company.estimated_revenue < 100000000:  # < $100M
                return "Growth Equity"
            else:
                return "Buyout"
        
        return None
    
    def _map_company_stage_to_pe_stage(self, company_stage: str) -> Optional[str]:
        """Map company stage to PE investment stage preference."""
        if not company_stage:
            return None
            
        stage = company_stage.lower()
        
        if any(word in stage for word in ["seed", "startup"]):
            return "seed"
        elif any(word in stage for word in ["early", "series_a", "series_b"]):
            return "early_stage"
        elif any(word in stage for word in ["growth", "expansion", "series_c"]):
            return "growth"
        elif any(word in stage for word in ["late", "mature", "established"]):
            return "late_stage"
        elif any(word in stage for word in ["buyout", "acquisition"]):
            return "buyout"
        
        return "growth"  # Default
    
    def _estimate_check_size_range(self, company: SMBCompanyData) -> tuple[Optional[float], Optional[float]]:
        """Estimate appropriate check size range based on company characteristics."""
        if not company.estimated_revenue:
            return None, None
        
        # Rule of thumb: investment size typically 0.1x to 2x annual revenue
        revenue_millions = company.estimated_revenue / 1000000
        
        min_check = max(1.0, revenue_millions * 0.1)  # At least $1M
        max_check = revenue_millions * 2.0
        
        # Cap based on company stage
        stage = company.growth_stage.lower() if company.growth_stage else ""
        
        if any(word in stage for word in ["seed", "startup"]):
            max_check = min(max_check, 10.0)  # Cap at $10M for early stage
        elif any(word in stage for word in ["early", "growth"]):
            max_check = min(max_check, 50.0)  # Cap at $50M for growth
        
        return min_check, max_check
    
    def _map_company_size_to_target(self, company: SMBCompanyData) -> Optional[str]:
        """Map company size to PE target company size category."""
        if not company.estimated_revenue:
            return None
        
        revenue_millions = company.estimated_revenue / 1000000
        
        if revenue_millions < 10:
            return "small"
        elif revenue_millions < 100:
            return "medium"
        elif revenue_millions < 1000:
            return "large"
        else:
            return "enterprise"
    
    def _enhance_synthetic_matches(
        self, 
        matches: List[Dict[str, Any]], 
        company: SMBCompanyData
    ) -> List[Dict[str, Any]]:
        """Enhance synthetic fund matches with additional reasoning."""
        enhanced = []
        
        for match in matches:
            enhanced_match = match.copy()
            enhanced_match["source"] = "synthetic_pe_fund"
            enhanced_match["match_type"] = "PE Fund"
            
            # Add confidence scoring
            confidence_score = self._calculate_synthetic_confidence(match, company)
            enhanced_match["confidence_score"] = confidence_score
            
            # Add investment fit reasoning
            fit_reasoning = self._analyze_investment_fit(match.get("fund"), company)
            enhanced_match["investment_fit"] = fit_reasoning
            
            enhanced.append(enhanced_match)
        
        return enhanced
    
    def _enhance_adv_match(
        self, 
        match: Dict[str, Any], 
        company: SMBCompanyData
    ) -> Dict[str, Any]:
        """Enhance ADV firm match with additional reasoning."""
        enhanced_match = match.copy()
        enhanced_match["source"] = "adv_sec_filing"
        enhanced_match["match_type"] = "Investment Advisory Firm"
        
        firm = match.get("firm", {})
        
        # Calculate match confidence
        confidence_score = self._calculate_adv_confidence(match, company)
        enhanced_match["confidence_score"] = confidence_score
        
        # Analyze potential fit
        fit_analysis = self._analyze_adv_fit(firm, company)
        enhanced_match["potential_fit"] = fit_analysis
        
        # Add firm characteristics
        enhanced_match["firm_characteristics"] = {
            "total_assets": firm.get("total_assets"),
            "employee_range": firm.get("employee_range"),
            "legal_structure": firm.get("legal_structure"),
            "services": self._extract_adv_services(firm),
            "location": f"{firm.get('address_city', '')}, {firm.get('address_state', '')}"
        }
        
        return enhanced_match
    
    def _calculate_synthetic_confidence(
        self, 
        match: Dict[str, Any], 
        company: SMBCompanyData
    ) -> float:
        """Calculate confidence score for synthetic fund match."""
        fund = match.get("fund", {})
        if not fund:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Industry alignment
        if company.primary_industry.lower() in fund.get("focus_sector", "").lower():
            confidence += 0.3
        
        # Stage alignment  
        if company.growth_stage.lower() in fund.get("focus_stage", "").lower():
            confidence += 0.2
        
        # Adjust by semantic distance (lower distance = higher confidence)
        distance = match.get("distance", 1.0)
        distance_score = max(0, 1.0 - distance)
        confidence += distance_score * 0.3
        
        return min(1.0, confidence)
    
    def _calculate_adv_confidence(
        self, 
        match: Dict[str, Any], 
        company: SMBCompanyData
    ) -> float:
        """Calculate confidence score for ADV firm match."""
        firm = match.get("firm", {})
        if not firm:
            return 0.0
        
        confidence = 0.3  # Lower base confidence for ADV (less targeted)
        
        # Asset size alignment
        if firm.get("total_assets") and company.estimated_revenue:
            asset_ratio = firm["total_assets"] / company.estimated_revenue
            if 0.1 <= asset_ratio <= 100:  # Reasonable range
                confidence += 0.2
        
        # Relevant services
        if firm.get("provides_investment_advice"):
            confidence += 0.1
        if firm.get("strategy_private_funds"):
            confidence += 0.2
        if firm.get("strategy_equity"):
            confidence += 0.1
        
        # Semantic similarity
        score = match.get("score", 1.0)
        semantic_score = max(0, 1.0 - score)
        confidence += semantic_score * 0.2
        
        return min(1.0, confidence)
    
    def _analyze_investment_fit(
        self, 
        fund: Optional[Dict[str, Any]], 
        company: SMBCompanyData
    ) -> Dict[str, str]:
        """Analyze investment fit between fund and company."""
        if not fund:
            return {"overall": "No fund data available"}
        
        analysis = {}
        
        # Sector fit
        if company.primary_industry.lower() in fund.get("focus_sector", "").lower():
            analysis["sector"] = "Strong sector alignment"
        else:
            analysis["sector"] = f"Sector mismatch: Fund focuses on {fund.get('focus_sector')}"
        
        # Stage fit
        if company.growth_stage.lower() in fund.get("focus_stage", "").lower():
            analysis["stage"] = "Good stage fit"
        else:
            analysis["stage"] = f"Stage consideration: Fund targets {fund.get('focus_stage')}"
        
        # Size fit (rough estimate)
        fund_aum = fund.get("aum_musd", 0)
        if company.estimated_revenue and fund_aum > 0:
            if fund_aum >= company.estimated_revenue / 1000000:  # Fund AUM > company revenue
                analysis["size"] = "Fund size appropriate for investment"
            else:
                analysis["size"] = "Fund may be too small for this investment"
        
        return analysis
    
    def _analyze_adv_fit(
        self, 
        firm: Dict[str, Any], 
        company: SMBCompanyData
    ) -> Dict[str, str]:
        """Analyze potential fit between ADV firm and company."""
        analysis = {}
        
        # Service alignment
        services = []
        if firm.get("provides_investment_advice"):
            services.append("Investment Advisory")
        if firm.get("provides_financial_planning"):
            services.append("Financial Planning")
        if firm.get("provides_pension_consulting"):
            services.append("Pension Consulting")
        
        if services:
            analysis["services"] = f"Offers: {', '.join(services)}"
        else:
            analysis["services"] = "Limited service information available"
        
        # Investment strategy alignment
        strategies = []
        if firm.get("strategy_private_funds"):
            strategies.append("Private Funds")
        if firm.get("strategy_equity"):
            strategies.append("Equity")
        if firm.get("strategy_hedge_funds"):
            strategies.append("Hedge Funds")
        
        if strategies:
            analysis["strategies"] = f"Investment strategies: {', '.join(strategies)}"
        else:
            analysis["strategies"] = "General investment advisory services"
        
        return analysis
    
    def _extract_adv_services(self, firm: Dict[str, Any]) -> List[str]:
        """Extract list of services from ADV firm data."""
        services = []
        
        if firm.get("provides_investment_advice"):
            services.append("Investment Advice")
        if firm.get("provides_financial_planning"):
            services.append("Financial Planning")
        if firm.get("provides_pension_consulting"):
            services.append("Pension Consulting")
        
        return services
    
    def _create_combined_ranking(
        self,
        synthetic_matches: List[Dict[str, Any]],
        adv_matches: List[Dict[str, Any]],
        total_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Create a combined ranking of all matches."""
        all_matches = []
        
        # Add synthetic matches with source weighting
        for match in synthetic_matches:
            match_copy = match.copy()
            # Weight synthetic matches higher (more targeted for PE)
            match_copy["weighted_score"] = match.get("confidence_score", 0) * 1.2
            all_matches.append(match_copy)
        
        # Add ADV matches with source weighting
        for match in adv_matches:
            match_copy = match.copy()
            # Weight ADV matches normally
            match_copy["weighted_score"] = match.get("confidence_score", 0) * 1.0
            all_matches.append(match_copy)
        
        # Sort by weighted score (descending)
        all_matches.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        
        # Limit results if specified
        if total_k:
            all_matches = all_matches[:total_k]
        
        return all_matches
    
    def _generate_match_summary(
        self,
        synthetic_matches: List[Dict[str, Any]],
        adv_matches: List[Dict[str, Any]],
        company: SMBCompanyData
    ) -> Dict[str, Any]:
        """Generate a summary of all matches."""
        summary = {
            "total_matches": len(synthetic_matches) + len(adv_matches),
            "synthetic_fund_count": len(synthetic_matches),
            "adv_firm_count": len(adv_matches),
            "top_match_source": None,
            "coverage_analysis": {}
        }
        
        # Determine top match source
        if synthetic_matches or adv_matches:
            all_matches = synthetic_matches + adv_matches
            top_match = max(all_matches, key=lambda x: x.get("confidence_score", 0))
            summary["top_match_source"] = top_match.get("source")
        
        # Coverage analysis
        if synthetic_matches:
            avg_confidence = sum(m.get("confidence_score", 0) for m in synthetic_matches) / len(synthetic_matches)
            summary["coverage_analysis"]["synthetic_avg_confidence"] = avg_confidence
        
        if adv_matches:
            avg_confidence = sum(m.get("confidence_score", 0) for m in adv_matches) / len(adv_matches)
            summary["coverage_analysis"]["adv_avg_confidence"] = avg_confidence
        
        return summary


def main():
    """Command-line interface for enhanced fund matching."""
    import argparse
    import json
    from api.agents.smb_agent import SMBCompanyData, CompanySize, ConfidenceLevel
    
    parser = argparse.ArgumentParser(description="Enhanced fund matching demo")
    parser.add_argument("--company-name", required=True, help="Company name")
    parser.add_argument("--industry", required=True, help="Primary industry")
    parser.add_argument("--stage", default="growth", help="Company stage")
    parser.add_argument("--revenue", type=float, help="Estimated revenue (USD)")
    parser.add_argument("--synthetic-k", type=int, default=5, help="Number of synthetic matches")
    parser.add_argument("--adv-k", type=int, default=10, help="Number of ADV matches")
    parser.add_argument("--total-k", type=int, help="Total results limit")
    
    args = parser.parse_args()
    
    # Create sample company data using SMBCompanyData
    company = SMBCompanyData(
        company_name=args.company_name,
        website_url="https://example.com",
        business_description=f"{args.company_name} operates in {args.industry}",
        primary_industry=args.industry,
        industry_keywords=[args.industry.split()[-1]],
        naics_codes=[],
        target_customers="B2B",
        estimated_employees=50,
        estimated_revenue=args.revenue,
        size_band=CompanySize.MEDIUM,
        growth_stage=args.stage,
        headquarters_location="Unknown",
        service_areas=[],
        business_model="Unknown",
        revenue_streams=[],
        key_products_services=[],
        competitive_advantages=[],
        growth_indicators=[],
        technology_assets=[],
        customer_base_description="Unknown",
        scalability_factors=[],
        market_position="Unknown",
        defensibility=[],
        profitability_indicators=[],
        capital_requirements="Unknown",
        confidence_scores={
            "company_identification": ConfidenceLevel.HIGH,
            "size_estimation": ConfidenceLevel.MEDIUM
        },
        data_sources=["CLI Input"],
        analysis_notes=[],
        input_tokens=0,
        output_tokens=0,
        estimated_cost_usd=0.0
    )
    
    # Initialize enhanced matcher
    matcher = EnhancedFundMatcher()
    
    # Perform matching
    results = matcher.match_company(
        company=company,
        k_synthetic=args.synthetic_k,
        k_adv=args.adv_k,
        total_k=args.total_k
    )
    
    # Display results
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()