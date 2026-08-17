"""Two-stage CrossEncoder reranking over Hybrid candidates only."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sentence_transformers import CrossEncoder

from .config import DEFAULT_RERANK_TOP_K, MODEL_CACHE_DIR, RERANKER_MODEL_NAME
from .hybrid_retriever import HybridRetrievedSong


@dataclass(frozen=True, slots=True)
class RerankedSong(HybridRetrievedSong):
    """Hybrid candidate with CrossEncoder relevance and final rank."""

    reranker_score: float
    final_rank: int


class CrossEncoderReranker:
    """Small explicit wrapper around sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        *,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        if model is None:
            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            model = CrossEncoder(model_name, cache_folder=str(MODEL_CACHE_DIR))
        self.model = model

    def rerank(
        self,
        query: str,
        candidates: list[HybridRetrievedSong],
        top_k: int = DEFAULT_RERANK_TOP_K,
    ) -> list[RerankedSong]:
        """Score query/document pairs without introducing any new song IDs."""

        if not query.strip():
            raise ValueError("query must be a non-empty string.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not candidates:
            return []

        pairs = [(query, candidate.document_text) for candidate in candidates]
        raw_scores = self.model.predict(pairs)
        if len(raw_scores) != len(candidates):
            raise ValueError("CrossEncoder score count must match candidate count.")

        scored: list[tuple[HybridRetrievedSong, float, int]] = []
        for original_rank, (candidate, raw_score) in enumerate(
            zip(candidates, raw_scores, strict=True), start=1
        ):
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError(f"CrossEncoder returned a non-finite score: {score}")
            scored.append((candidate, score, original_rank))

        scored.sort(key=lambda item: (-item[1], item[2], item[0].song_id))
        results = []
        seen_ids: set[str] = set()
        for candidate, reranker_score, _ in scored:
            if candidate.song_id in seen_ids:
                continue
            seen_ids.add(candidate.song_id)
            results.append(
                RerankedSong(
                    song_id=candidate.song_id,
                    title=candidate.title,
                    artist=candidate.artist,
                    album=candidate.album,
                    release_year=candidate.release_year,
                    genres=list(candidate.genres),
                    tags=list(candidate.tags),
                    score=reranker_score,
                    document_text=candidate.document_text,
                    dense_rank=candidate.dense_rank,
                    dense_score=candidate.dense_score,
                    bm25_rank=candidate.bm25_rank,
                    bm25_score=candidate.bm25_score,
                    fusion_score=candidate.fusion_score,
                    reranker_score=reranker_score,
                    final_rank=len(results) + 1,
                )
            )
            if len(results) == top_k:
                break
        return results


def rerank(
    query: str,
    candidates: list[HybridRetrievedSong],
    top_k: int = DEFAULT_RERANK_TOP_K,
    *,
    reranker: CrossEncoderReranker | None = None,
) -> list[RerankedSong]:
    """Convenience interface for Hybrid candidate reranking."""

    active_reranker = reranker or CrossEncoderReranker()
    return active_reranker.rerank(query, candidates, top_k)
