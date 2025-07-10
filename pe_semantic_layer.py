import os
import sqlite3
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions
from sqlalchemy import create_engine, MetaData, Table, select
import logging

from data_models import ProcessedCompanyData

# Disable ChromaDB anonymous telemetry to avoid console noise
os.environ.setdefault("CHROMA_TELEMETRY", "0")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)

class SemanticFirmSearcher:
    """Semantic search layer on top of the synthetic SEC ADV SQLite database.

    This helper uses ChromaDB as a vector store for fast semantic look-up of
    private-equity-relevant firm records that live in the `sec_adv_synthetic.db`
    SQLite database.  The workflow is:

    1.  Build / load a persistent Chroma collection (default: ``pe_firms``).
    2.  Execute semantic search over company descriptions.
    3.  Resolve matched ids back to the relational database via SQLAlchemy.
    4.  (Optional)  Pass the raw firm row to the Gemini-powered research agent
        to obtain rich, structured PE intelligence.
    """

    DEFAULT_COLLECTION = "pe_firms"

    def __init__(
        self,
        db_path: str = "sec_adv_synthetic.db",
        persist_dir: str = ".chroma_pe_firms",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.db_path = db_path
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        # ---------- set-up SQLAlchemy  ---------- #
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.metadata = MetaData()
        self.firms_table: Table = Table("firms", self.metadata, autoload_with=self.engine)

        # ---------- set-up ChromaDB  ---------- #
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # Load or (re)build collection
        if self.DEFAULT_COLLECTION in [c.name for c in self.client.list_collections()]:
            self.collection = self.client.get_collection(
                self.DEFAULT_COLLECTION, embedding_function=self.embedding_fn
            )
        else:
            # Build the collection from scratch
            self.collection = self.client.create_collection(
                self.DEFAULT_COLLECTION, embedding_function=self.embedding_fn
            )
            self._populate_collection()

        # ---------- research agent ---------- #
        # Research agent functionality archived - CompanyResearchAgent moved to archive/legacy_agents/
        # self._research_agent: Optional[CompanyResearchAgent] = None  # lazy-init

    # --------------------------------------------------------------------- #
    #                         Collection population                         #
    # --------------------------------------------------------------------- #
    def _populate_collection(self) -> None:
        """Read all firm rows and embed them into the Chroma collection."""

        conn = self.engine.connect()
        stmt = select(self.firms_table)
        rows = conn.execute(stmt).fetchall()
        conn.close()

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for row in rows:
            # Each row is a sqlalchemy Row object -> map to dict via _mapping
            data = dict(row._mapping)
            doc_str = self._row_to_document(data)
            ids.append(str(data["id"]))
            documents.append(doc_str)
            metadatas.append({"table": "firms", "row_id": data["id"]})

        # Chroma expects batches of <= 100k. Split if needed (DB <100k rows typical).
        CHUNK = 5000
        for start in range(0, len(ids), CHUNK):
            end = start + CHUNK
            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    @staticmethod
    def _row_to_document(data: Dict[str, Any]) -> str:
        """Convert a firm DB row into an unstructured text blob for embedding."""
        parts = [
            f"Firm Name: {data.get('firm_name', '')}",
            f"Address: {data.get('address', '')}",
            f"Legal Structure: {data.get('legal_structure', '')}",
        ]
        if data.get("crd_number"):
            parts.append(f"CRD Number: {data['crd_number']}")
        if data.get("cik_number"):
            parts.append(f"CIK Number: {data['cik_number']}")
        return " | ".join(parts)

    # --------------------------------------------------------------------- #
    #                               Search                                 #
    # --------------------------------------------------------------------- #
    def semantic_search(
        self, query: str, n_results: int = 10, enrich: bool = False
    ) -> List[Dict[str, Any]]:
        """Perform a semantic search and optionally enrich with LLM analysis.

        Returns list of dictionaries with keys: `score`, `db_row`, and (if
        ``enrich`` is True) `llm_analysis` (instance of ProcessedCompanyData).
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["distances", "metadatas"],
        )

        # results is a dict. For single query, take index 0
        ids: List[str] = results["ids"][0]
        distances: List[float] = results["distances"][0]

        output: List[Dict[str, Any]] = []

        # Fetch DB rows in a single query
        if ids:
            conn = self.engine.connect()
            stmt = select(self.firms_table).where(self.firms_table.c.id.in_(ids))
            db_rows = {str(row.id): dict(row._mapping) for row in conn.execute(stmt)}
            conn.close()
        else:
            db_rows = {}

        for idx, row_id in enumerate(ids):
            db_row = db_rows.get(row_id)
            item: Dict[str, Any] = {
                "score": distances[idx],  # smaller is better with Euclidean
                "db_row": db_row,
            }
            if enrich and db_row:
                # Research agent functionality has been archived
                # Skip LLM enrichment as CompanyResearchAgent is no longer available
                item["llm_analysis_error"] = "LLM enrichment disabled - research agent archived"
                output.append(item)
                continue

                try:
                    import asyncio

                    coro = self._research_agent.research_company(db_row["firm_name"])
                    analysis: ProcessedCompanyData = asyncio.run(coro)
                    item["llm_analysis"] = analysis
                except Exception as exc:
                    item["llm_analysis_error"] = str(exc)
            output.append(item)

        return output


# ------------------------------------------------------------------------- #
#                      Convenience script entry-point                      #
# ------------------------------------------------------------------------- #

def main():
    import argparse, json

    parser = argparse.ArgumentParser(description="Semantic search PE firms")
    parser.add_argument("query", help="Search query, e.g. 'healthcare services'")
    parser.add_argument("--k", type=int, default=10, help="Number of results")
    parser.add_argument("--enrich", action="store_true", help="Run LLM enrichment")
    args = parser.parse_args()

    searcher = SemanticFirmSearcher()
    results = searcher.semantic_search(args.query, n_results=args.k, enrich=args.enrich)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main() 