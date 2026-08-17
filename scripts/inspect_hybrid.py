"""Inspect dense, BM25, and filtered RRF rankings for fixed Phase 5 queries."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag.hybrid_retriever import retrieve_hybrid_with_details  # noqa: E402


QUERIES = [
    "melodic 80s hard rock power ballad",
    "fast aggressive thrash metal",
    "progressive melodic metal from the 1990s",
    "classic traditional heavy metal from the 1980s",
]


def print_basic_ranking(label: str, songs: list, score_label: str) -> None:
    print(f"\n=== {label} ===")
    for rank, song in enumerate(songs, 1):
        print(
            f"{rank}. {song.title} — {song.artist} | year={song.release_year} "
            f"| {score_label}={song.score:.6f}"
        )


def main() -> None:
    for query in QUERIES:
        result = retrieve_hybrid_with_details(query, top_k=10)
        print(f"\n\nQuery: {query}")
        print(f"Constraints: {result.constraints}")
        print_basic_ranking("Dense", result.dense, "similarity")
        print_basic_ranking("BM25", result.bm25, "BM25 score")

        print("\n=== Hybrid + Metadata ===")
        for rank, song in enumerate(result.hybrid, 1):
            print(
                f"{rank}. {song.title} — {song.artist} | year={song.release_year} "
                f"| dense_rank={song.dense_rank} | bm25_rank={song.bm25_rank} "
                f"| fusion_score={song.fusion_score:.6f}"
            )

        dense_ids = [song.song_id for song in result.dense]
        hybrid_ids = [song.song_id for song in result.hybrid]
        same_ids = set(dense_ids) & set(hybrid_ids)
        moved_up = [
            song_id
            for song_id in same_ids
            if hybrid_ids.index(song_id) < dense_ids.index(song_id)
        ]
        moved_down = [
            song_id
            for song_id in same_ids
            if hybrid_ids.index(song_id) > dense_ids.index(song_id)
        ]
        newly_introduced = [song_id for song_id in hybrid_ids if song_id not in dense_ids]
        print("\n=== Comparison ===")
        print(f"Same IDs: {len(same_ids)} / 10")
        print(f"Moved up: {moved_up}")
        print(f"Moved down: {moved_down}")
        print(f"Filtered by metadata: {result.filtered_song_ids}")
        print(f"Newly introduced: {newly_introduced}")
        print(
            "Latency: "
            f"embedding={result.timing.query_embedding_seconds:.6f}s, "
            f"Milvus={result.timing.milvus_search_seconds:.6f}s, "
            f"BM25={result.timing.bm25_seconds:.6f}s, "
            f"fusion/filter={result.timing.fusion_and_filter_seconds:.6f}s"
        )


if __name__ == "__main__":
    main()
