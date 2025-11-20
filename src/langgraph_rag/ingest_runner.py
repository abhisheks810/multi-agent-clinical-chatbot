# src/langgraph_rag/ingest_runner.py

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any

import mlflow
import pandas as pd

from .config import IngestionConfig, TableConfig
from .chunking import build_chunked_documents
from .vectorstore import get_or_create_chroma_collection, index_chunks


def _config_to_dict(config: IngestionConfig) -> Dict[str, Any]:
    return {
        "tables": [
            {
                "name": t.name,
                "path": t.path,
                "id_column": t.id_column,
                "text_columns": t.text_columns,
            }
            for t in config.tables
        ],
        "metadata_text_path": config.metadata_text_path,
        "persist_dir": config.persist_dir,
        "collection_name": config.collection_name,
        "embedding_model": config.embedding_model,
        "chunk_token_size": config.chunk_token_size,
        "chunk_overlap": config.chunk_overlap,
    }


def run_ingestion_with_mlflow(
    config: IngestionConfig,
    *,
    reset: bool = False,
    experiment_name: str = "vectorstore_ingestion",
):
    # 1) Select experiment
    mlflow.set_experiment(experiment_name)

    cfg_dict = _config_to_dict(config)

    # 2) Start run
    with mlflow.start_run(run_name="build_vector_store") as run:
        run_id = run.info.run_id

        # ---- Log params ----
        mlflow.log_params({
            "persist_dir": config.persist_dir,
            "collection_name": config.collection_name,
            "embedding_model": config.embedding_model,
            "chunk_token_size": config.chunk_token_size,
            "chunk_overlap": config.chunk_overlap,
            "reset": reset,
        })
        # per-table params
        for t in config.tables:
            mlflow.log_param(f"table_{t.name}_path", t.path)
            mlflow.log_param(f"table_{t.name}_id_column", t.id_column)
            mlflow.log_param(f"table_{t.name}_text_columns", t.text_columns or "AUTO")

        # Save full config as artifact
        artifacts_dir = Path("mlflow_artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        (artifacts_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))
        mlflow.log_artifact(str(artifacts_dir / "config.json"), artifact_path="config")

        # ---- Data loading & basic stats ----
        t0 = time.time()
        table_row_counts = {}

        for t in config.tables:
            df = pd.read_csv(t.path, sep="\t")
            table_row_counts[t.name] = len(df)
            mlflow.log_metric(f"n_rows_{t.name}", len(df))

        t_load = time.time()

        # ---- Chunking ----
        chunks = build_chunked_documents(config)
        t_chunk = time.time()

        n_chunks = len(chunks)
        mlflow.log_metric("n_chunks_total", n_chunks)

        if n_chunks > 0:
            avg_len = sum(len(c.text) for c in chunks) / n_chunks
            mlflow.log_metric("avg_chunk_length_chars", avg_len)

        # log a few sample chunks
        sample = [
            {"id": c.id, "text": c.text[:300], "metadata": c.metadata}
            for c in chunks[:5]
        ]
        (artifacts_dir / "sample_chunks.json").write_text(json.dumps(sample, indent=2))
        mlflow.log_artifact(str(artifacts_dir / "sample_chunks.json"),
                            artifact_path="samples")

        # ---- Vector store creation & indexing ----
        collection = get_or_create_chroma_collection(config)
        t_vs_create = time.time()

        index_chunks(chunks, collection)  # batched in your updated code
        t_index = time.time()

        # collection count metric
        mlflow.log_metric("collection_count", collection.count())

        # ---- Timing metrics ----
        mlflow.log_metric("time_load_data_sec", t_load - t0)
        mlflow.log_metric("time_chunking_sec", t_chunk - t_load)
        mlflow.log_metric("time_collection_create_sec", t_vs_create - t_chunk)
        mlflow.log_metric("time_indexing_sec", t_index - t_vs_create)
        mlflow.log_metric("time_total_sec", t_index - t0)

        return collection, run_id
