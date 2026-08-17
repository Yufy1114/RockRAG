"""Retrieve real catalog candidates, then generate grounded recommendations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag.config import (  # noqa: E402
    DEFAULT_RECOMMENDATION_COUNT,
    GENERATION_RETRIEVAL_TOP_K,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OLLAMA_MODEL,
)
from rockrag.generator import (  # noqa: E402
    GenerationUnavailableError,
    generate_recommendations,
)
from rockrag.retriever import retrieve_with_timing  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve catalog songs and generate grounded recommendations."
    )
    parser.add_argument("query", nargs="?", help="Natural-language music request")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = args.query or input("Query:\n").strip()
    if not query:
        raise SystemExit("Query must not be empty.")

    total_started = perf_counter()
    candidates, retrieval_timing = retrieve_with_timing(
        query,
        top_k=GENERATION_RETRIEVAL_TOP_K,
    )

    print(f"\nQuery:\n{query}")
    print("\n=== Retrieved Candidates ===")
    for rank, candidate in enumerate(candidates, start=1):
        print(f"\n{rank}. {candidate.title} — {candidate.artist}")
        print(f"   similarity score: {candidate.score:.6f}")
        print(f"   year: {candidate.release_year}")
        print(f"   genres: {candidate.genres}")
        print(f"   tags: {candidate.tags}")

    generation_started = perf_counter()
    try:
        recommendations = generate_recommendations(
            query,
            candidates,
            recommendation_count=DEFAULT_RECOMMENDATION_COUNT,
        )
    except GenerationUnavailableError as exc:
        print(f"\n=== Final Recommendations ===\n\n{exc}")
        print("\n=== Latency ===")
        print(
            f"Query embedding: {retrieval_timing.query_embedding_seconds:.6f} seconds"
        )
        print(f"Milvus search: {retrieval_timing.milvus_search_seconds:.6f} seconds")
        print("LLM generation: unavailable")
        print(f"Total pipeline: {perf_counter() - total_started:.6f} seconds")
        return
    generation_seconds = perf_counter() - generation_started

    active_model = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else GEMINI_MODEL
    print(f"\n=== Final Recommendations ({LLM_PROVIDER}: {active_model}) ===")
    for rank, recommendation in enumerate(recommendations, start=1):
        print(f"\n{rank}. {recommendation.title} — {recommendation.artist}")
        print(f"   Reason: {recommendation.reason}")

    print("\n=== Latency ===")
    print(f"Query embedding: {retrieval_timing.query_embedding_seconds:.6f} seconds")
    print(f"Milvus search: {retrieval_timing.milvus_search_seconds:.6f} seconds")
    print(f"LLM generation: {generation_seconds:.6f} seconds")
    print(f"Total pipeline: {perf_counter() - total_started:.6f} seconds")


if __name__ == "__main__":
    main()
