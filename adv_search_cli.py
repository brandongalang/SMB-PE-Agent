"""
Improved CLI for ADV semantic search with clear build/search separation.

This provides a cleaner interface with separate commands for building the
ChromaDB collection and searching it.
"""

import argparse
import json
import sys
import logging
from typing import Optional

from adv_semantic_layer import AdvSemanticSearcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def confirm_build(limit: int) -> bool:
    """Ask user to confirm building the ChromaDB collection."""
    print(f"\n🏗️  ChromaDB collection needs to be built first.")
    print(f"📊 This will process {limit:,} firms and create vector embeddings.")
    print(f"⏱️  Estimated time: ~{estimate_build_time(limit)} minutes")
    print(f"💾 Memory usage: ~{estimate_memory_usage(limit)}MB")
    
    while True:
        response = input(f"\nProceed with building collection? [y/N]: ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no', '']:
            return False
        else:
            print("Please enter 'y' or 'n'")

def estimate_build_time(limit: int) -> int:
    """Estimate build time based on number of firms."""
    # Rough estimate: 5000 firms = 2 minutes
    return max(1, (limit * 2) // 5000)

def estimate_memory_usage(limit: int) -> int:
    """Estimate memory usage in MB."""
    # Rough estimate: 5000 firms = 200MB
    return (limit * 200) // 5000

def check_collection_exists(searcher: AdvSemanticSearcher) -> bool:
    """Check if ChromaDB collection exists and has data."""
    try:
        stats = searcher.get_stats()
        return stats.get("chromadb_collection_count", 0) > 0
    except Exception:
        return False

def cmd_build(args):
    """Build the ChromaDB collection."""
    print(f"🏗️  Building ChromaDB collection with {args.limit:,} firms...")
    
    # Temporarily update the limit in the source if needed
    # (This is a bit hacky, but works for the demo)
    
    searcher = AdvSemanticSearcher(db_url=args.db_url)
    
    if not args.force and check_collection_exists(searcher):
        stats = searcher.get_stats()
        current_count = stats.get("chromadb_collection_count", 0)
        print(f"⚠️  Collection already exists with {current_count:,} firms.")
        print("Use --force to rebuild anyway.")
        return
    
    if not args.yes and not confirm_build(args.limit):
        print("❌ Build cancelled.")
        return
    
    # Update the limit by temporarily modifying the query
    # (In a production system, you'd pass this as a parameter)
    import adv_semantic_layer
    original_populate = adv_semantic_layer.AdvSemanticSearcher._populate_collection
    
    def limited_populate(self):
        """Modified populate function with custom limit."""
        logging.info("Populating ChromaDB collection with ADV firm data...")
        
        session = self.SessionLocal()
        try:
            # Get firms with custom limit
            firms = session.query(adv_semantic_layer.AdvFirm).filter(
                adv_semantic_layer.AdvFirm.searchable_text.isnot(None)
            ).limit(args.limit).all()
            
            if not firms:
                logging.warning("No firms found with searchable text. Run the processing script first.")
                return
            
            logging.info(f"Found {len(firms)} firms to add to ChromaDB collection.")
            
            ids = []
            documents = []
            metadatas = []
            
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
    
    # Temporarily replace the method
    adv_semantic_layer.AdvSemanticSearcher._populate_collection = limited_populate
    
    try:
        searcher.rebuild_collection()
        print(f"✅ Successfully built ChromaDB collection with {args.limit:,} firms!")
        
        # Show final stats
        stats = searcher.get_stats()
        print(f"📊 Collection size: {stats.get('chromadb_collection_count', 0):,} firms")
        
    finally:
        # Restore original method
        adv_semantic_layer.AdvSemanticSearcher._populate_collection = original_populate

def cmd_search(args):
    """Search the ChromaDB collection."""
    searcher = AdvSemanticSearcher(db_url=args.db_url)
    
    # Check if collection exists
    if not check_collection_exists(searcher):
        print("❌ ChromaDB collection not found!")
        print("💡 Run 'python adv_search_cli.py build' first to create the collection.")
        sys.exit(1)
    
    # Get collection stats
    stats = searcher.get_stats()
    collection_size = stats.get("chromadb_collection_count", 0)
    
    print(f"🔍 Searching {collection_size:,} firms for: '{args.query}'")
    
    # Build filters if provided
    filters = {}
    if args.legal_structure:
        filters["legal_structure"] = args.legal_structure
    if args.state:
        filters["address_state"] = args.state
    
    # Perform search
    results = searcher.semantic_search(
        query=args.query,
        n_results=args.k,
        filters=filters if filters else None
    )
    
    # Display results
    if not results:
        print("❌ No results found.")
        return
    
    print(f"\n📋 Found {len(results)} results:")
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
        
        if args.verbose and result.get("searchable_text"):
            print(f"   Context: {result['searchable_text'][:200]}...")
        print()

def cmd_stats(args):
    """Show database and collection statistics."""
    searcher = AdvSemanticSearcher(db_url=args.db_url)
    
    try:
        stats = searcher.get_stats()
        
        print("📊 Database Statistics")
        print("=" * 50)
        print(f"Total firms in database: {stats.get('total_firms', 0):,}")
        print(f"Firms with searchable text: {stats.get('firms_with_searchable_text', 0):,}")
        print(f"ChromaDB collection size: {stats.get('chromadb_collection_count', 0):,}")
        
        print("\n🏛️ Legal Structures:")
        legal_structures = stats.get('legal_structures', {})
        for structure, count in sorted(legal_structures.items(), key=lambda x: x[1], reverse=True)[:10]:
            if structure:  # Skip null values
                print(f"  {structure}: {count:,}")
        
        print("\n🗺️ Top States:")
        top_states = stats.get('top_states', {})
        for state, count in sorted(top_states.items(), key=lambda x: x[1], reverse=True)[:10]:
            if state and state != 'null':  # Skip null values
                print(f"  {state}: {count:,}")
                
        # Collection status
        collection_size = stats.get('chromadb_collection_count', 0)
        if collection_size > 0:
            print(f"\n✅ ChromaDB collection ready for searching")
        else:
            print(f"\n❌ ChromaDB collection not built yet")
            print(f"💡 Run 'python adv_search_cli.py build' to create it")
            
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")

def main():
    """Main CLI function with subcommands."""
    parser = argparse.ArgumentParser(
        description="ADV Semantic Search CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build collection with 20,000 firms
  python adv_search_cli.py build --limit 20000
  
  # Search for healthcare firms
  python adv_search_cli.py search "healthcare investment" --k 5
  
  # Search with filters
  python adv_search_cli.py search "technology" --state CA --legal-structure Corporation
  
  # Show statistics
  python adv_search_cli.py stats
        """
    )
    
    # Global arguments
    parser.add_argument("--db-url", default="sqlite:///adv_database.db", help="Database URL")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build ChromaDB collection")
    build_parser.add_argument("--limit", type=int, default=20000, help="Number of firms to process (default: 20000)")
    build_parser.add_argument("--force", action="store_true", help="Force rebuild even if collection exists")
    build_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search the collection")
    search_parser.add_argument("query", help="Search query (e.g., 'healthcare investment')")
    search_parser.add_argument("--k", type=int, default=10, help="Number of results to return")
    search_parser.add_argument("--legal-structure", help="Filter by legal structure")
    search_parser.add_argument("--state", help="Filter by state")
    search_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed context")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command
    if args.command == "build":
        cmd_build(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)

if __name__ == "__main__":
    main()