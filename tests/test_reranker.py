from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag.generator import build_candidate_context  # noqa: E402
from rockrag.hybrid_retriever import (  # noqa: E402
    HybridRetrievedSong,
    QueryConstraints,
    matches_hard_constraints,
)
from rockrag.reranker import CrossEncoderReranker  # noqa: E402


def hybrid(song_id: str, year: int = 1986) -> HybridRetrievedSong:
    return HybridRetrievedSong(
        song_id=song_id,
        title=f"Title {song_id}",
        artist="Artist",
        album=None,
        release_year=year,
        genres=["thrash metal"],
        tags=["aggressive"],
        score=0.03,
        document_text=f"Title: Title {song_id}\nArtist: Artist",
        dense_rank=1,
        dense_score=0.7,
        bm25_rank=2,
        bm25_score=3.0,
        fusion_score=0.03,
    )


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = np.asarray(scores, dtype=np.float32)
        self.pairs: list[tuple[str, str]] | None = None

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        self.pairs = pairs
        return self.scores


class RerankerTests(unittest.TestCase):
    def test_pairs_scores_order_and_candidate_subset(self) -> None:
        candidates = [hybrid("a"), hybrid("b"), hybrid("c")]
        model = FakeCrossEncoder([0.1, 0.9, 0.5])
        results = CrossEncoderReranker(model=model).rerank(
            "fast thrash metal", candidates, top_k=2
        )
        self.assertEqual(
            model.pairs,
            [("fast thrash metal", song.document_text) for song in candidates],
        )
        self.assertEqual([song.song_id for song in results], ["b", "c"])
        self.assertTrue(
            {song.song_id for song in results}.issubset(
                {song.song_id for song in candidates}
            )
        )
        self.assertTrue(all(math.isfinite(song.reranker_score) for song in results))
        self.assertEqual([song.final_rank for song in results], [1, 2])

    def test_deterministic_ties_and_no_duplicates(self) -> None:
        candidates = [hybrid("b"), hybrid("a"), hybrid("b")]
        results = CrossEncoderReranker(
            model=FakeCrossEncoder([0.5, 0.5, 0.5])
        ).rerank("metal", candidates, top_k=3)
        self.assertEqual([song.song_id for song in results], ["b", "a"])

    def test_hard_year_constraint_remains_true_after_reranking(self) -> None:
        constraints = QueryConstraints(year_min=1990, year_max=1999)
        candidates = [hybrid("a", 1992), hybrid("b", 1998)]
        results = CrossEncoderReranker(model=FakeCrossEncoder([0.1, 0.9])).rerank(
            "1990s metal", candidates
        )
        self.assertTrue(
            all(matches_hard_constraints(song, constraints) for song in results)
        )

    def test_hybrid_and_reranker_score_labels(self) -> None:
        candidate = hybrid("a")
        self.assertIn("RRF fusion score: 0.030000", build_candidate_context([candidate]))
        reranked = CrossEncoderReranker(model=FakeCrossEncoder([1.25])).rerank(
            "metal", [candidate]
        )[0]
        context = build_candidate_context([reranked])
        self.assertIn("Retrieval method: cross_encoder_reranker", context)
        self.assertIn("Reranker relevance score: 1.250000", context)


@unittest.skipUnless(
    os.environ.get("ROCKRAG_RUN_RERANKER_TESTS") == "1",
    "set ROCKRAG_RUN_RERANKER_TESTS=1 to run the cached real CrossEncoder",
)
class CrossEncoderIntegrationTests(unittest.TestCase):
    def test_real_cross_encoder_returns_finite_candidate_score(self) -> None:
        candidate = hybrid("a")
        result = CrossEncoderReranker().rerank("thrash metal", [candidate], 1)
        self.assertEqual([song.song_id for song in result], ["a"])
        self.assertTrue(math.isfinite(result[0].reranker_score))


if __name__ == "__main__":
    unittest.main()
