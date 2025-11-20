# src/langgraph_rag/vectorstore.py

from __future__ import annotations

from pathlib import Path
from typing import List

import os

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions  # <-- important
from dotenv import load_dotenv

from .config import IngestionConfig
from .chunking import DocChunk, build_chunked_documents

# Make sure environment variables from .env are loaded (OPENAI_API_KEY)
load_dotenv()


# ---------- Embedding function (OpenAI via Chroma helper) ----------

def get_embedding_function(config: IngestionConfig):
    """
    Returns a Chroma-compatible embedding function object.
    This object has __call__ + name(), which Chroma 2.x expects.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it in your environment or .env file."
        )

    # Chroma's built-in OpenAIEmbeddingFunction
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=config.embedding_model,
    )


# ---------- Chroma client + collection ----------

def get_or_create_chroma_collection(config: IngestionConfig) -> chromadb.api.models.Collection.Collection:
    """
    Returns a persistent Chroma collection for this project.
    """
    persist_dir = Path(config.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embedding_fn = get_embedding_function(config)

    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(allow_reset=False),
    )

    collection = client.get_or_create_collection(
        name=config.collection_name,
        embedding_function=embedding_fn,
    )

    return collection


# ---------- Index chunks ----------

def index_chunks(
    chunks: List[DocChunk],
    collection: chromadb.api.models.Collection.Collection,
    batch_size: int = 100,  # you can tune this
) -> None:
    """
    Adds chunks to the given Chroma collection in batches, so we don't exceed
    OpenAI's max tokens per request.
    """
    if not chunks:
        return

    n = len(chunks)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = chunks[start:end]

        ids = [c.id for c in batch]
        texts = [c.text for c in batch]
        metadatas = [c.metadata for c in batch]

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )



# ---------- High-level orchestration ----------

def build_persistent_vector_store(
    config: IngestionConfig,
    *,
    reset: bool = False,
) -> chromadb.api.models.Collection.Collection:
    """
    High-level helper:
      - builds chunked docs (TSV + metadata)
      - opens Chroma persistent collection
      - indexes chunks

    Returns the Chroma collection.
    """
    chunks = build_chunked_documents(config)
    collection = get_or_create_chroma_collection(config)
    index_chunks(chunks, collection)
    return collection
