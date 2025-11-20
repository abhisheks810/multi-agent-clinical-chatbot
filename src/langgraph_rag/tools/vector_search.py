# src/langgraph_rag/tools/vector_search.py

from __future__ import annotations

from typing import List, Dict

from langgraph_rag.config import IngestionConfig, TableConfig
from langgraph_rag.vectorstore import get_or_create_chroma_collection


def get_default_ingestion_config() -> IngestionConfig:
    """
    Default config pointing at your clinical TSV + metadata.

    Adjust paths / names here if needed.
    """
    return IngestionConfig(
        tables=[
            TableConfig(
                name="patients",
                path="data/patients.tsv",
                id_column="PATIENT_ID",  # adjust if your column differs
                text_columns=None,       # let chunking infer/concat text columns
            )
        ],
        metadata_text_path="data/clinical_patient_meta.txt",
        persist_dir="chroma_db",
        collection_name="patient_docs",
        # embedding_model is defined inside your existing IngestionConfig
    )


def get_patient_collection(config: IngestionConfig):
    """
    Open existing Chroma collection (must be already built).
    Does NOT rebuild or reset.
    """
    collection = get_or_create_chroma_collection(config)
    return collection


def vector_search_patients(
    config: IngestionConfig,
    query: str,
    k: int = 5,
) -> List[Dict]:
    """
    Query the Chroma collection for top-k relevant chunks.
    Returns a list of JSON-serializable dicts with snippet and metadata.
    """
    collection = get_patient_collection(config)

    res = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    results: List[Dict] = []
    for doc, meta, dist in zip(docs, metas, dists):
        results.append(
            {
                "id": meta.get("row_id") or meta.get("patient_id") or "UNKNOWN",
                "snippet": doc,
                "metadata": meta,
                "distance": float(dist),
            }
        )

    return results
