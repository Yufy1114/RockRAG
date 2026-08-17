"""Build the complete local Milvus Lite song collection from the catalog."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag import embed_documents, load_catalog, song_to_document  # noqa: E402
from rockrag.config import (  # noqa: E402
    MILVUS_COLLECTION_NAME,
    MILVUS_DATABASE_PATH,
    MILVUS_EMBEDDING_DIMENSION,
    MILVUS_INDEX_TYPE,
    MILVUS_METRIC_TYPE,
)
from rockrag.vector_store import SongVectorStore  # noqa: E402


def main() -> None:
    started = perf_counter()
    songs = load_catalog()
    print(f"Loaded {len(songs)} songs")

    documents = [song_to_document(song) for song in songs]
    embeddings = embed_documents(documents)
    print(f"Generated embeddings: {embeddings.shape}")

    print(f"\nCreating Milvus collection: {MILVUS_COLLECTION_NAME}")
    print(f"Embedding dimension: {MILVUS_EMBEDDING_DIMENSION}")
    print(f"Metric: {MILVUS_METRIC_TYPE}")
    print(f"Index: {MILVUS_INDEX_TYPE}")

    store = SongVectorStore()
    try:
        inserted = store.rebuild(songs, documents, embeddings)
        entity_count = store.count_entities()
        print(f"\nInserted: {inserted} entities")
        print(f"Collection entity count: {entity_count}")
        print(f"Database: {MILVUS_DATABASE_PATH}")

        if not len(songs) == embeddings.shape[0] == entity_count:
            raise RuntimeError(
                "Integrity failure: catalog, embedding, and Milvus counts differ."
            )

        print("\nStored metadata samples:")
        for entity in store.sample_entities(limit=3):
            print(
                f"- {entity['song_id']} | {entity['title']} — {entity['artist']}"
            )
            print(entity["document_text"])
    finally:
        store.close()

    print(f"\nBuild time: {perf_counter() - started:.6f} seconds")


if __name__ == "__main__":
    main()
