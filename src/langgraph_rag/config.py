# src/langgraph_rag/config.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TableConfig:
    """
    Configuration for one patient-related table (TSV).

    - name: logical name (e.g. "demographics", "treatments", "labs")
    - path: path to TSV file
    - id_column: column that links to patient_id (or episode_id etc.)
    - text_columns: if None/empty -> auto-infer from available columns
    """
    name: str
    path: str
    id_column: str = "patient_id"
    text_columns: Optional[List[str]] = None


@dataclass
class IngestionConfig:
    """
    Global ingestion config for vector store creation.
    You can have 1 or many tables in `tables`.
    """
    tables: List[TableConfig]
    metadata_text_path: str 

    # Chroma persistence
    persist_dir: str = "chroma_db"
    collection_name: str = "patient_docs"

    # Embeddings
    embedding_model: str = "text-embedding-3-large"

    # Chunking
    chunk_token_size: int = 512
    chunk_overlap: int = 64
