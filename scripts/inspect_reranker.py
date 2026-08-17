"""Compare Hybrid Top-10 with CrossEncoder-reranked Top-10."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag.hybrid_retriever import retrieve_hybrid  # noqa: E402
from rockrag.reranker import CrossEncoderReranker  # noqa: E402


QUERIES = [
    "melodic 80s hard rock power ballad",
    "fast aggressive thrash metal",
    "progressive melodic metal from the 1990s",
    "classic traditional heavy metal from the 1980s",
]


def main() -> None:
    reranker = CrossEncoderReranker()
    for query in QUERIES:
        hybrid = retrieve_hybrid(query, top_k=10)
        started = perf_counter()
        reranked = reranker.rerank(query, hybrid, top_k=10)
        elapsed = perf_counter() - started

        print(f"\n\nQuery: {query}")
        print("\n=== Hybrid Top-10 ===")
        for rank, song in enumerate(hybrid, 1):
            print(
                f"{rank}. {song.title} — {song.artist} | "
                f"RRF={song.fusion_score:.6f}"
            )
        print("\n=== Reranked Top-10 ===")
        for song in reranked:
            hybrid_rank = next(
                rank for rank, item in enumerate(hybrid, 1) if item.song_id == song.song_id
            )
            print(
                f"{song.final_rank}. {song.title} — {song.artist} | "
                f"hybrid_rank={hybrid_rank} | RRF={song.fusion_score:.6f} | "
                f"reranker={song.reranker_score:.6f}"
            )
        print(f"Reranker latency: {elapsed:.6f}s")


if __name__ == "__main__":
    main()
