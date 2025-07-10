import os
# Disable Chroma telemetry
os.environ.setdefault("CHROMA_TELEMETRY", "0")
import logging
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from sqlalchemy import create_engine, MetaData, Table, select

# from data_models import ProcessedCompanyData  # DEPRECATED - using SMBCompanyData instead
from api.agents.smb_agent import SMBCompanyData


class SemanticFundSearcher:
    """Vector search over the `pe_funds` table."""

    DEFAULT_COLLECTION = "pe_funds"

    def __init__(
        self,
        db_path: str = "sec_adv_synthetic.db",
        persist_dir: str = ".chroma_pe_funds",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.db_path = db_path
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.metadata = MetaData()
        self.funds_table: Table = Table("pe_funds", self.metadata, autoload_with=self.engine)

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        if self.DEFAULT_COLLECTION in [c.name for c in self.client.list_collections()]:
            self.collection = self.client.get_collection(
                self.DEFAULT_COLLECTION, embedding_function=self.embedding_fn
            )
        else:
            self.collection = self.client.create_collection(
                self.DEFAULT_COLLECTION, embedding_function=self.embedding_fn
            )
            self._populate_collection()

    def _populate_collection(self):
        conn = self.engine.connect()
        rows = conn.execute(select(self.funds_table)).fetchall()
        conn.close()
        ids, docs, metas = [], [], []
        for r in rows:
            d = dict(r._mapping)
            ids.append(str(d["id"]))
            doc = self._row_to_document(d)
            docs.append(doc)
            metas.append({"fund_id": d["id"]})
        if ids:
            self.collection.add(ids=ids, documents=docs, metadatas=metas)

    @staticmethod
    def _row_to_document(data: Dict[str, Any]) -> str:
        return (
            f"{data['fund_name']} | {data['strategy']} | {data['focus_sector']} | "
            f"{data['focus_stage']} | {data['focus_geo']} | AUM {data['aum_musd']}M | {data['thesis']}"
        )

    # --------------------------------------------------------------
    def query(self, text: str, k: int = 10) -> List[Dict[str, Any]]:
        """Return list with distance and DB row."""
        res = self.collection.query(
            query_texts=[text], n_results=k, include=["distances", "metadatas"]
        )
        ids = res["ids"][0]
        dists = res["distances"][0]
        if not ids:
            return []
        conn = self.engine.connect()
        db_rows = {
            str(r.id): dict(r._mapping)
            for r in conn.execute(
                select(self.funds_table).where(self.funds_table.c.id.in_(ids))
            )
        }
        conn.close()
        return [
            {"distance": dists[i], "fund": db_rows.get(fid)} for i, fid in enumerate(ids)
        ]


class FundMatcher:
    """High-level interface: SMB profile -> ranked funds with basic reasoning."""

    def __init__(self):
        self.searcher = SemanticFundSearcher()

    def _build_query(self, smb: SMBCompanyData) -> str:
        parts = [smb.primary_industry]
        parts.extend(smb.industry_keywords[:5])
        if smb.size_band:
            parts.append(smb.size_band.name)
        if smb.growth_stage:  # Changed from company_stage
            parts.append(smb.growth_stage)
        return " ".join(filter(None, parts))

    def match(self, smb: SMBCompanyData, k: int = 10) -> List[Dict[str, Any]]:
        query = self._build_query(smb)
        raw = self.searcher.query(query, k)
        results = []
        for item in raw:
            fund = item["fund"]
            reason_parts = []
            # simple heuristic overlaps
            if fund and smb.primary_industry.lower() in fund["focus_sector"].lower():
                reason_parts.append("Sector match")
            if smb.growth_stage.lower() in fund["focus_stage"].lower():  # Changed from company_stage
                reason_parts.append("Stage fit")
            results.append({
                **item,
                "match_reason": ", ".join(reason_parts) if reason_parts else "Semantic similarity"
            })
        return results 