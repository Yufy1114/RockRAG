"""Semantic retrieval from Milvus without generation or ranking layers."""

from __future__ import annotations

import json
import math
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_TOP_K, MILVUS_DATABASE_PATH
from .embeddings import embed_query
from .vector_store import SongVectorStore


@dataclass(frozen=True, slots=True)
class RetrievedSong:
    song_id: str
    title: str
    artist: str
    album: str | None
    release_year: int | None
    genres: list[str]
    tags: list[str]
    score: float
    document_text: str


@dataclass(frozen=True, slots=True)
class RetrievalTiming:
    query_embedding_seconds: float
    milvus_search_seconds: float


def map_milvus_hit(hit: dict[str, Any]) -> RetrievedSong:
    """Convert one raw Milvus hit into the project's retrieval structure."""

    entity = hit["entity"]
    score = float(hit["distance"])
    if not math.isfinite(score):
        raise ValueError(f"Milvus returned a non-finite similarity score: {score}")

    return RetrievedSong(
        # PyMilvus 3 exposes a VARCHAR primary key under its field name.
        song_id=str(hit.get("song_id", hit.get("id"))),
        title=entity["title"],
        artist=entity["artist"],
        album=entity.get("album"),
        release_year=entity.get("release_year"),
        genres=json.loads(entity["genres_json"]),
        tags=json.loads(entity["tags_json"]),
        score=score,
        document_text=entity["document_text"],
    )


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    database_path: str | Path = MILVUS_DATABASE_PATH,
) -> list[RetrievedSong]:
    """Embed one natural-language query and map its Milvus COSINE hits."""

    results, _ = retrieve_with_timing(
        query,
        top_k,
        database_path=database_path,
    )
    return results


def retrieve_with_timing(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    database_path: str | Path = MILVUS_DATABASE_PATH,
) -> tuple[list[RetrievedSong], RetrievalTiming]:
    """Retrieve songs and expose simple embedding/search latency measurements."""

    embedding_started = perf_counter()
    query_embedding = embed_query(query)
    embedding_seconds = perf_counter() - embedding_started
    store = SongVectorStore(database_path=database_path)
    try:
        search_started = perf_counter()
        hits = store.search(query_embedding, top_k=top_k)
        search_seconds = perf_counter() - search_started
        return (
            [map_milvus_hit(hit) for hit in hits],
            RetrievalTiming(
                query_embedding_seconds=embedding_seconds,
                milvus_search_seconds=search_seconds,
            ),
        )
    finally:
        store.close()
