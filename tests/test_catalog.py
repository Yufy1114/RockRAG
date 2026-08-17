from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag import CatalogValidationError, SongRecord, load_catalog, song_to_document


class CatalogTests(unittest.TestCase):
    def test_project_catalog_loads_with_unique_ids(self) -> None:
        songs = load_catalog()
        self.assertGreaterEqual(len(songs), 30)
        self.assertEqual(len(songs), len({song.song_id for song in songs}))

    def test_duplicate_song_id_has_clear_error(self) -> None:
        record = {
            "song_id": "duplicate",
            "title": "Example",
            "artist": "Example Artist",
            "album": None,
            "release_year": None,
            "genres": [],
            "tags": [],
            "musicbrainz_recording_id": None,
            "musicbrainz_release_group_id": None,
            "isrc": None,
            "metadata_source": "test",
            "tags_source": None,
            "verified_at": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "songs.json"
            path.write_text(json.dumps([record, record]), encoding="utf-8")
            with self.assertRaisesRegex(CatalogValidationError, "duplicate song_id"):
                load_catalog(path)

    def test_release_year_must_be_integer(self) -> None:
        record = {
            "song_id": "bad-year",
            "title": "Example",
            "artist": "Example Artist",
            "album": None,
            "release_year": "1984",
            "genres": [],
            "tags": [],
            "musicbrainz_recording_id": None,
            "musicbrainz_release_group_id": None,
            "isrc": None,
            "metadata_source": "test",
            "tags_source": None,
            "verified_at": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "songs.json"
            path.write_text(json.dumps([record]), encoding="utf-8")
            with self.assertRaisesRegex(CatalogValidationError, "release_year"):
                load_catalog(path)

    def test_song_to_document_has_fixed_order_and_omits_missing_fields(self) -> None:
        song = SongRecord(
            song_id="example",
            title="Still Loving You",
            artist="Scorpions",
            release_year=1984,
            genres=["hard rock"],
            tags=["power ballad", "melodic"],
        )
        self.assertEqual(
            song_to_document(song),
            "Title: Still Loving You\n"
            "Artist: Scorpions\n"
            "Release year: 1984\n"
            "Genres: hard rock\n"
            "Tags: power ballad, melodic",
        )


if __name__ == "__main__":
    unittest.main()
