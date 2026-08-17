from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag.generator import (  # noqa: E402
    RecommendationItem,
    RecommendationResponse,
    validate_recommendations,
)
from rockrag.hybrid_retriever import (  # noqa: E402
    HybridRetrievedSong,
    QueryConstraints,
    matches_hard_constraints,
    parse_query_constraints,
    reciprocal_rank_fusion,
    retrieve_bm25,
    retrieve_hybrid_with_details,
)
from rockrag.models import SongRecord  # noqa: E402
from rockrag.retriever import RetrievedSong, RetrievalTiming  # noqa: E402


def retrieved(song_id: str, year: int | None = 1985) -> RetrievedSong:
    return RetrievedSong(
        song_id=song_id,
        title=song_id,
        artist="Artist",
        album=None,
        release_year=year,
        genres=["thrash metal"],
        tags=["aggressive"],
        score=0.5,
        document_text=f"Title: {song_id}",
    )


class HybridRetrieverTests(unittest.TestCase):
    def test_1980s_constraint(self) -> None:
        for query in ("80s hard rock", "1980s metal"):
            constraints = parse_query_constraints(query)
            self.assertEqual((constraints.year_min, constraints.year_max), (1980, 1989))

    def test_1990s_constraint(self) -> None:
        for query in ("90s progressive metal", "from the 1990s"):
            constraints = parse_query_constraints(query)
            self.assertEqual((constraints.year_min, constraints.year_max), (1990, 1999))

    def test_explicit_artist_constraint(self) -> None:
        constraints = parse_query_constraints(
            "thrash metal by Slayer", known_artists=["Slayer", "Anthrax"]
        )
        self.assertEqual(constraints.artist, "Slayer")

    def test_year_filter_excludes_outside_and_null(self) -> None:
        constraints = QueryConstraints(year_min=1990, year_max=1999)
        self.assertTrue(matches_hard_constraints(retrieved("in", 1992), constraints))
        self.assertFalse(matches_hard_constraints(retrieved("old", 1989), constraints))
        self.assertFalse(matches_hard_constraints(retrieved("missing", None), constraints))

    def test_bm25_scores_are_finite_unique_and_catalog_is_unchanged(self) -> None:
        songs = [
            SongRecord("a", "Raining Blood", "Slayer", release_year=1986, tags=["thrash metal", "aggressive"]),
            SongRecord("b", "Ballad", "Artist", release_year=1987, tags=["power ballad"]),
        ]
        before = [song.to_dict() for song in songs]
        results = retrieve_bm25("aggressive thrash metal", 2, songs=songs)
        self.assertEqual(results[0].song_id, "a")
        self.assertTrue(all(math.isfinite(song.score) for song in results))
        self.assertEqual(len({song.song_id for song in results}), len(results))
        self.assertEqual([song.to_dict() for song in songs], before)

    def test_rrf_is_deterministic_and_deduplicates_within_ranking(self) -> None:
        a, b = retrieved("a"), retrieved("b")
        first = reciprocal_rank_fusion([[a, a, b], [b, a]], rrf_constant=60)
        second = reciprocal_rank_fusion([[a, a, b], [b, a]], rrf_constant=60)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"a", "b"})

    def test_hybrid_result_is_unique_and_limited_to_top_k(self) -> None:
        songs = [
            SongRecord("a", "A", "Artist", release_year=1985),
            SongRecord("b", "B", "Artist", release_year=1986),
            SongRecord("c", "C", "Artist", release_year=1987),
        ]
        dense = [retrieved("a"), retrieved("b"), retrieved("c")]
        bm25 = [retrieved("c"), retrieved("b"), retrieved("a")]
        with (
            patch("rockrag.hybrid_retriever.load_catalog", return_value=songs),
            patch(
                "rockrag.hybrid_retriever.retrieve_with_timing",
                return_value=(dense, RetrievalTiming(0.0, 0.0)),
            ),
            patch("rockrag.hybrid_retriever.retrieve_bm25", return_value=bm25),
        ):
            result = retrieve_hybrid_with_details("metal", top_k=2)
        self.assertLessEqual(len(result.hybrid), 2)
        self.assertEqual(
            len({song.song_id for song in result.hybrid}), len(result.hybrid)
        )

    def test_generator_consumes_hybrid_candidate(self) -> None:
        source = retrieved("a", 1992)
        candidate = HybridRetrievedSong(
            song_id=source.song_id,
            title=source.title,
            artist=source.artist,
            album=source.album,
            release_year=source.release_year,
            genres=source.genres,
            tags=source.tags,
            score=0.03,
            document_text=source.document_text,
            dense_rank=2,
            dense_score=0.7,
            bm25_rank=1,
            bm25_score=3.0,
            fusion_score=0.03,
        )
        response = RecommendationResponse(
            recommendations=[RecommendationItem(song_id="a", reason="Visible tag")]
        )
        result = validate_recommendations(response, [candidate], 5)
        self.assertEqual(result[0].song_id, "a")


if __name__ == "__main__":
    unittest.main()
