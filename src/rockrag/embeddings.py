"""Thin BGE embedding functions used by RockRAG."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import MODEL_CACHE_DIR


MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model: SentenceTransformer | None = None


def load_model() -> SentenceTransformer:
    """Load BGE once and return the shared model instance."""

    global _model
    if _model is None:
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _model = SentenceTransformer(
            MODEL_NAME,
            cache_folder=str(MODEL_CACHE_DIR),
        )
        dimension = _model.get_embedding_dimension()
        if dimension != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Expected {EMBEDDING_DIMENSION}-dimensional embeddings from "
                f"{MODEL_NAME}, but the model reports {dimension}."
            )
    return _model


def embed_documents(texts: list[str]) -> np.ndarray:
    """Encode document text without a query prefix as normalized rows."""

    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Document texts must be non-empty strings.")

    embeddings = load_model().encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings)


def embed_query(query: str) -> np.ndarray:
    """Encode one query with BGE's retrieval instruction and L2 normalization."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    # This prefix is part of BGE retrieval encoding. Documents do not use it.
    instructed_query = QUERY_INSTRUCTION + query.strip()
    embedding = load_model().encode(
        instructed_query,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embedding)
