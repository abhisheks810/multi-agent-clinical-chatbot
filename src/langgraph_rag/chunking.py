# src/langgraph_rag/chunking.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import tiktoken

from .config import IngestionConfig, TableConfig


@dataclass
class DocChunk:
    id: str
    text: str
    metadata: Dict[str, Any]


# ---------- Tokenisation + chunking ----------

def get_token_encoder(model: str = "cl100k_base"):
    return tiktoken.get_encoding(model)


def chunk_text(
    text: str,
    *,
    encoder,
    chunk_token_size: int,
    chunk_overlap: int,
) -> List[str]:
    tokens = encoder.encode(text)
    chunks: List[str] = []

    start = 0
    n = len(tokens)

    while start < n:
        end = min(start + chunk_token_size, n)
        chunk_tokens = tokens[start:end]
        chunk = encoder.decode(chunk_tokens)
        chunks.append(chunk)

        if end == n:
            break

        start = end - chunk_overlap

    return chunks


# ---------- Column inference ----------

def infer_text_columns(df: pd.DataFrame, table_cfg: TableConfig) -> List[str]:
    """
    If text_columns is explicitly set on the table → use that.
    Otherwise:
      - use all columns except the id_column
      - optionally you can filter to 'object'/string columns if you want
    """
    if table_cfg.text_columns:
        return table_cfg.text_columns

    non_id_cols = [c for c in df.columns if c != table_cfg.id_column]
    # You can choose to keep all non-id cols (including numeric) or only object cols.
    # For now, let's keep all non-id columns.
    # If you want only string-like columns:
    # object_cols = [c for c in non_id_cols if df[c].dtype == "object"]
    # return object_cols or non_id_cols
    return non_id_cols


# ---------- Load any table as big text docs ----------

def _load_table_rows_as_text(
    table_cfg: TableConfig,
) -> List[Dict[str, Any]]:
    """
    Returns list of dicts for a single table:
    {
        "id": <row_key>,   # usually patient_id
        "text": <combined text>,
        "metadata": {...}
    }
    """
    path = Path(table_cfg.path)
    if not path.exists():
        raise FileNotFoundError(f"TSV file not found: {path}")

    df = pd.read_csv(path, sep="\t")

    if table_cfg.id_column not in df.columns:
        raise ValueError(
            f"id_column='{table_cfg.id_column}' not in columns for table '{table_cfg.name}': "
            f"{df.columns.tolist()}"
        )

    text_cols = infer_text_columns(df, table_cfg)

    docs: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        row_id = str(row[table_cfg.id_column])

        parts = []
        for col in text_cols:
            val = row[col]
            if pd.isna(val):
                continue
            parts.append(f"{col}: {val}")

        combined = "\n".join(parts)

        docs.append(
            {
                "id": row_id,  # you can prefix by table name if you want
                "text": combined,
                "metadata": {
                    "source": "msk_chord",
                    "table_name": table_cfg.name,
                    "id_column": table_cfg.id_column,
                    "row_id": row_id,
                    "row_index": int(idx),
                },
            }
        )

    return docs


def load_metadata_text(
    metadata_text_path: str,
) -> Dict[str, Any]:
    path = Path(metadata_text_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata text file not found: {path}")

    text = path.read_text(encoding="utf-8")

    return {
        "id": path.name,
        "text": text,
        "metadata": {
            "source": "metadata_file",
            "filename": path.name,
        },
    }


# ---------- High-level: build chunked docs from ALL tables ----------

def build_chunked_documents(
    config: IngestionConfig,
) -> List[DocChunk]:
    """
    - For each table in config.tables:
        - loads its TSV
        - auto-selects text columns (if none specified)
        - builds per-row textual docs
        - chunks them
    - Also chunks the single metadata text file
    - Returns a flat list[DocChunk]
    """
    encoder = get_token_encoder()
    all_chunks: List[DocChunk] = []

    # 1) Patient-related tables
    for table_cfg in config.tables:
        base_docs = _load_table_rows_as_text(table_cfg)
        for doc in base_docs:
            base_id = doc["id"]
            base_meta = doc["metadata"]
            text = doc["text"]

            text_chunks = chunk_text(
                text,
                encoder=encoder,
                chunk_token_size=config.chunk_token_size,
                chunk_overlap=config.chunk_overlap,
            )

            row_index = base_meta["row_index"]

            for i, chunk in enumerate(text_chunks):
                all_chunks.append(
                    DocChunk(
                        id=f"{table_cfg.name}:{base_id}:{row_index}:chunk_{i}",
                        text=chunk,
                        metadata={**base_meta, "chunk_index": i},
                    )
                )

    # 2) Global metadata file
    metadata_doc = load_metadata_text(config.metadata_text_path)
    meta_chunks = chunk_text(
        metadata_doc["text"],
        encoder=encoder,
        chunk_token_size=config.chunk_token_size,
        chunk_overlap=config.chunk_overlap,
    )
    for i, chunk in enumerate(meta_chunks):
        all_chunks.append(
            DocChunk(
                id=f"{metadata_doc['id']}_chunk_{i}",
                text=chunk,
                metadata={
                    **metadata_doc["metadata"],
                    "chunk_index": i,
                },
            )
        )

    return all_chunks
