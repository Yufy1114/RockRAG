"""Explicit Milvus Lite storage operations for song vectors."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from pymilvus import DataType, MilvusClient

from .config import (
    MILVUS_COLLECTION_NAME,
    MILVUS_DATABASE_PATH,
    MILVUS_EMBEDDING_DIMENSION,
    MILVUS_INDEX_TYPE,
    MILVUS_METRIC_TYPE,
)
from .models import SongRecord


SEARCH_OUTPUT_FIELDS = [
    "title",
    "artist",
    "album",
    "release_year",
    "genres_json",
    "tags_json",
    "document_text",
]


class SongVectorStore:
    """Small MilvusClient wrapper that keeps every database step visible."""

    def __init__(
        self,
        database_path: str | Path = MILVUS_DATABASE_PATH,
        collection_name: str = MILVUS_COLLECTION_NAME,
    ) -> None:
        self.database_path = Path(database_path)
        self.collection_name = collection_name
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = MilvusClient(uri=str(self.database_path))
        if self.client.has_collection(self.collection_name):
            self.client.load_collection(self.collection_name)

    @staticmethod
    def build_schema() -> Any:
        """Return the explicit schema used by the songs collection."""

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="song_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=256,
        )
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=MILVUS_EMBEDDING_DIMENSION,
        )
        schema.add_field(
            field_name="title", datatype=DataType.VARCHAR, max_length=512
        )
        schema.add_field(
            field_name="artist", datatype=DataType.VARCHAR, max_length=512
        )
        schema.add_field(
            field_name="album",
            datatype=DataType.VARCHAR,
            max_length=1024,
            nullable=True,
        )
        schema.add_field(
            field_name="release_year", datatype=DataType.INT64, nullable=True
        )
        schema.add_field(
            field_name="genres_json", datatype=DataType.VARCHAR, max_length=4096
        )
        schema.add_field(
            field_name="tags_json", datatype=DataType.VARCHAR, max_length=4096
        )
        schema.add_field(
            field_name="document_text", datatype=DataType.VARCHAR, max_length=8192
        )
        return schema

    @staticmethod
    def build_index_params() -> Any:
        """Return an exact FLAT cosine index for the embedding field."""

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_name="embedding_flat_cosine",
            index_type=MILVUS_INDEX_TYPE,
            metric_type=MILVUS_METRIC_TYPE,
        )
        return index_params

    def create_collection(self, *, drop_existing: bool = True) -> None:
        """Drop the old collection when requested, then create a fresh one."""

        if self.client.has_collection(self.collection_name):
            if not drop_existing:
                raise ValueError(
                    f"Collection '{self.collection_name}' already exists."
                )
            self.client.drop_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=self.build_schema(),
            index_params=self.build_index_params(),
        )
        self.client.load_collection(self.collection_name)

    @staticmethod
    def _entities_from_songs(
        songs: list[SongRecord],
        documents: list[str],
        embeddings: np.ndarray,
    ) -> list[dict[str, Any]]:
        if len(songs) != len(documents) or len(songs) != embeddings.shape[0]:
            raise ValueError(
                "Song, document, and embedding counts must be identical."
            )
        if embeddings.ndim != 2 or embeddings.shape[1] != MILVUS_EMBEDDING_DIMENSION:
            raise ValueError(
                "Embeddings must have shape "
                f"(n, {MILVUS_EMBEDDING_DIMENSION}); got {embeddings.shape}."
            )

        entities = []
        for song, document, embedding in zip(
            songs, documents, embeddings, strict=True
        ):
            entities.append(
                {
                    "song_id": song.song_id,
                    "embedding": embedding.astype(np.float32).tolist(),
                    "title": song.title,
                    "artist": song.artist,
                    "album": song.album,
                    "release_year": song.release_year,
                    "genres_json": json.dumps(song.genres, ensure_ascii=False),
                    "tags_json": json.dumps(song.tags, ensure_ascii=False),
                    "document_text": document,
                }
            )
        return entities

    def insert_songs(
        self,
        songs: list[SongRecord],
        documents: list[str],
        embeddings: np.ndarray,
    ) -> int:
        """Insert aligned song metadata, documents, and vectors."""

        entities = self._entities_from_songs(songs, documents, embeddings)
        result = self.client.insert(
            collection_name=self.collection_name,
            data=entities,
        )
        self.client.flush(collection_name=self.collection_name)
        return int(result["insert_count"])

    def rebuild(
        self,
        songs: list[SongRecord],
        documents: list[str],
        embeddings: np.ndarray,
    ) -> int:
        """Drop, recreate, and repopulate the collection from scratch."""

        self.create_collection(drop_existing=True)
        return self.insert_songs(songs, documents, embeddings)

    def count_entities(self) -> int:
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats["row_count"])

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Run one COSINE search and return Milvus's raw hit dictionaries."""

        if query_embedding.shape != (MILVUS_EMBEDDING_DIMENSION,):
            raise ValueError(
                f"Query embedding must have shape ({MILVUS_EMBEDDING_DIMENSION},)."
            )
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        result = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding.astype(np.float32).tolist()],
            anns_field="embedding",
            limit=top_k,
            search_params={"metric_type": MILVUS_METRIC_TYPE, "params": {}},
            output_fields=SEARCH_OUTPUT_FIELDS,
        )
        return result[0]

    def sample_entities(
        self, limit: int = 3, *, seed: int = 42
    ) -> list[dict[str, Any]]:
        """Read a reproducible random sample for metadata integrity checks."""

        response = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["song_id", "title", "artist", "document_text"],
            limit=self.count_entities(),
        )
        entities = list(response)
        return random.Random(seed).sample(entities, k=min(limit, len(entities)))

    def describe_collection(self) -> dict[str, Any]:
        return self.client.describe_collection(self.collection_name)

    def close(self) -> None:
        self.client.close()
