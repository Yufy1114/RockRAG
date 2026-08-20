from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag.catalog_loader import load_catalog  # noqa: E402
from rockrag.tools import (  # noqa: E402
    build_playlist,
    compare_songs,
    get_song,
    search_songs,
)


def fake_ranked(song_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        song_id=song_id,
        title="Title",
        artist="Artist",
        album=None,
        release_year=1986,
        genres=["thrash metal"],
        tags=["aggressive"],
        reranker_score=1.0,
    )


class ToolTests(unittest.TestCase):
    def test_search_songs_returns_only_catalog_ids(self) -> None:
        catalog_ids = {song.song_id for song in load_catalog()}
        selected = next(iter(catalog_ids))
        with patch("rockrag.tools._retrieve_and_rerank", return_value=[fake_ranked(selected)]):
            result = search_songs("metal", 1)
        self.assertTrue({song["song_id"] for song in result}.issubset(catalog_ids))

    def test_build_playlist_respects_count(self) -> None:
        songs = [fake_ranked("a"), fake_ranked("b")]
        with patch("rockrag.tools._retrieve_and_rerank", return_value=songs) as pipeline:
            result = build_playlist("thrash", 2)
        self.assertEqual(len(result), 2)
        pipeline.assert_called_once_with("thrash", 2)

    def test_get_song_valid_and_missing(self) -> None:
        found = get_song("scorpions-still-loving-you")
        self.assertEqual(found["status"], "found")
        self.assertEqual(found["artist"], "Scorpions")
        self.assertEqual(get_song("missing-song"), {"status": "not_found", "song_id": "missing-song"})

    def test_compare_songs_shared_metadata(self) -> None:
        result = compare_songs([
            "scorpions-still-loving-you",
            "whitesnake-is-this-love",
        ])
        self.assertIn("hard rock", result["shared"]["shared_tags"])
        self.assertIn("power ballad", result["shared"]["shared_tags"])
        self.assertEqual(result["missing_song_ids"], [])

    def test_compare_songs_argument_count_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 2 and 5"):
            compare_songs(["one"])
        with self.assertRaisesRegex(ValueError, "between 2 and 5"):
            compare_songs(["1", "2", "3", "4", "5", "6"])

    def test_tools_do_not_mutate_catalog_file(self) -> None:
        path = PROJECT_ROOT / "data" / "processed" / "songs.json"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        get_song("scorpions-still-loving-you")
        compare_songs(["scorpions-still-loving-you", "whitesnake-is-this-love"])
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
