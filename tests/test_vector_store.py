from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag.config import MILVUS_EMBEDDING_DIMENSION  # noqa: E402
from rockrag.models import SongRecord  # noqa: E402
from rockrag.retriever import map_milvus_hit  # noqa: E402
from rockrag.vector_store import SongVectorStore  # noqa: E402


class VectorStoreUnitTests(unittest.TestCase):
    def test_schema_dimension_is_384(self) -> None:
        schema = SongVectorStore.build_schema().to_dict()
        embedding = next(
            field for field in schema["fields"] if field["name"] == "embedding"
        )
        self.assertEqual(embedding["params"]["dim"], 384)

    def test_retrieved_song_mapping(self) -> None:
        hit = {
            "song_id": "song-a",
            "distance": 0.75,
            "entity": {
                "title": "Song A",
                "artist": "Artist A",
                "album": None,
                "release_year": 1986,
                "genres_json": '["thrash metal"]',
                "tags_json": '["fast"]',
                "document_text": "Title: Song A\nArtist: Artist A",
            },
        }
        result = map_milvus_hit(hit)
        self.assertEqual(result.song_id, "song-a")
        self.assertEqual(result.genres, ["thrash metal"])
        self.assertEqual(result.tags, ["fast"])
        self.assertTrue(math.isfinite(result.score))


@unittest.skipUnless(
    os.environ.get("ROCKRAG_RUN_MILVUS_TESTS") == "1",
    "set ROCKRAG_RUN_MILVUS_TESTS=1 to allow Milvus Lite local port binding",
)
class MilvusLiteIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.store = SongVectorStore(
            Path(self.temp_directory.name) / "integration.db"
        )
        self.songs = [
            SongRecord(song_id="a", title="A", artist="Artist", tags=["first"]),
            SongRecord(song_id="b", title="B", artist="Artist", tags=["second"]),
            SongRecord(song_id="c", title="C", artist="Artist", tags=["third"]),
        ]
        self.documents = [f"Title: {song.title}" for song in self.songs]
        self.embeddings = np.zeros(
            (len(self.songs), MILVUS_EMBEDDING_DIMENSION), dtype=np.float32
        )
        self.embeddings[0, 0] = 1.0
        self.embeddings[1, 1] = 1.0
        self.embeddings[2, :2] = np.float32(1 / np.sqrt(2))

    def tearDown(self) -> None:
        self.store.close()
        self.temp_directory.cleanup()

    def test_repeated_rebuild_keeps_catalog_count(self) -> None:
        first_insert = self.store.rebuild(
            self.songs, self.documents, self.embeddings
        )
        second_insert = self.store.rebuild(
            self.songs, self.documents, self.embeddings
        )
        self.assertEqual(first_insert, len(self.songs))
        self.assertEqual(second_insert, len(self.songs))
        self.assertEqual(self.store.count_entities(), len(self.songs))

    def test_search_count_scores_mapping_and_numpy_ids(self) -> None:
        self.store.rebuild(self.songs, self.documents, self.embeddings)
        query = np.zeros(MILVUS_EMBEDDING_DIMENSION, dtype=np.float32)
        query[0] = 1.0

        hits = self.store.search(query, top_k=2)
        mapped = [map_milvus_hit(hit) for hit in hits]
        numpy_indices = np.argsort(self.embeddings @ query)[::-1][:2]
        numpy_ids = [self.songs[int(index)].song_id for index in numpy_indices]

        self.assertLessEqual(len(mapped), 2)
        self.assertTrue(all(math.isfinite(item.score) for item in mapped))
        self.assertEqual([item.song_id for item in mapped], numpy_ids)


if __name__ == "__main__":
    unittest.main()
