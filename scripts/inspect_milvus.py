"""Compare exact NumPy retrieval with Milvus Lite FLAT/COSINE search."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag import embed_documents, embed_query, load_catalog, song_to_document  # noqa: E402
from rockrag.config import DEFAULT_TOP_K  # noqa: E402
from rockrag.retriever import map_milvus_hit  # noqa: E402
from rockrag.vector_store import SongVectorStore  # noqa: E402


QUERIES = [
    "melodic 80s hard rock power ballad",
    "classic traditional heavy metal from the 1980s",
    "fast aggressive thrash metal",
    "progressive melodic metal from the 1990s",
]


def main() -> None:
    songs = load_catalog()
    documents = [song_to_document(song) for song in songs]
    song_embeddings = embed_documents(documents)
    store = SongVectorStore()
    retrieval_times: list[float] = []

    try:
        for query in QUERIES:
            query_embedding = embed_query(query)

            numpy_scores = song_embeddings @ query_embedding
            numpy_indices = np.argsort(numpy_scores)[::-1][:DEFAULT_TOP_K]
            numpy_ids = [songs[int(index)].song_id for index in numpy_indices]

            search_started = perf_counter()
            raw_hits = store.search(query_embedding, top_k=DEFAULT_TOP_K)
            retrieval_times.append(perf_counter() - search_started)
            milvus_results = [map_milvus_hit(hit) for hit in raw_hits]
            milvus_ids = [result.song_id for result in milvus_results]
            milvus_scores = np.array(
                [result.score for result in milvus_results], dtype=np.float32
            )
            ranked_numpy_scores = numpy_scores[numpy_indices]

            print(f"\nQuery: {query}")
            print("\nNumPy:")
            for rank, index in enumerate(numpy_indices, start=1):
                song = songs[int(index)]
                print(
                    f"{rank}. {song.title} — {song.artist} | "
                    f"{float(numpy_scores[index]):.6f}"
                )

            print("\nMilvus:")
            for rank, result in enumerate(milvus_results, start=1):
                print(
                    f"{rank}. {result.title} — {result.artist} | "
                    f"{result.score:.6f}"
                )
            print(
                f"Same IDs at same ranks: "
                f"{sum(a == b for a, b in zip(numpy_ids, milvus_ids, strict=True))} "
                f"/ {DEFAULT_TOP_K}"
            )
            print(
                f"Same ID set: {len(set(numpy_ids) & set(milvus_ids))} "
                f"/ {DEFAULT_TOP_K}"
            )
            print(
                "Maximum absolute score difference: "
                f"{float(np.max(np.abs(ranked_numpy_scores - milvus_scores))):.10f}"
            )
            print(f"Milvus search time: {retrieval_times[-1]:.6f} seconds")
    finally:
        store.close()

    print(
        f"\nMean single-query Milvus search time: "
        f"{float(np.mean(retrieval_times)):.6f} seconds"
    )


if __name__ == "__main__":
    main()
