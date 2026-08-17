"""Evaluate Dense, Hybrid, and Hybrid+CrossEncoder on identical qrels."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import metrics_for_ranking  # noqa: E402
from rockrag.catalog_loader import load_catalog  # noqa: E402
from rockrag.hybrid_retriever import retrieve_hybrid_with_details  # noqa: E402
from rockrag.reranker import CrossEncoderReranker  # noqa: E402
from rockrag.retriever import retrieve_with_timing  # noqa: E402


QUERY_PATH = PROJECT_ROOT / "evaluation" / "queries.json"
RESULT_PATH = PROJECT_ROOT / "evaluation" / "results" / "latest.json"
SYSTEMS = ("dense", "hybrid", "hybrid_reranker")


def load_queries(path: Path = QUERY_PATH) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")
    catalog_ids = {song.song_id for song in load_catalog()}
    seen_query_ids: set[str] = set()
    for item in data:
        required = {"query_id", "query", "rationale", "relevance"}
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError("Each evaluation item needs query_id/query/rationale/relevance.")
        if item["query_id"] in seen_query_ids:
            raise ValueError(f"Duplicate query_id: {item['query_id']}")
        seen_query_ids.add(item["query_id"])
        if not item["query"].strip() or not item["rationale"].strip():
            raise ValueError("Evaluation query and rationale must be non-empty.")
        if not item["relevance"]:
            raise ValueError(f"{item['query_id']} has no relevant songs.")
        for song_id, grade in item["relevance"].items():
            if song_id not in catalog_ids:
                raise ValueError(f"Unknown qrel song_id: {song_id}")
            if grade not in (1, 2):
                raise ValueError(f"Qrel grade must be 1 or 2: {song_id}={grade}")
    return data


def _ranking_payload(songs: list) -> list[dict]:
    payload = []
    for rank, song in enumerate(songs, 1):
        item = {
            "rank": rank,
            "song_id": song.song_id,
            "title": song.title,
            "artist": song.artist,
            "score": song.score,
        }
        for field in (
            "dense_rank",
            "dense_score",
            "bm25_rank",
            "bm25_score",
            "fusion_score",
            "reranker_score",
            "final_rank",
        ):
            if hasattr(song, field):
                item[field] = getattr(song, field)
        payload.append(item)
    return payload


def _mean_metrics(per_query: list[dict], system: str) -> dict[str, float]:
    keys = per_query[0]["metrics"][system]
    return {
        key: statistics.fmean(item["metrics"][system][key] for item in per_query)
        for key in keys
    }


def evaluate() -> dict:
    queries = load_queries()

    reranker_load_started = perf_counter()
    reranker = CrossEncoderReranker()
    reranker_load_seconds = perf_counter() - reranker_load_started

    # Warm BGE, Milvus, BM25, and CrossEncoder before measured iterations.
    warm_query = queries[0]["query"]
    retrieve_with_timing(warm_query, top_k=10)
    warm_hybrid = retrieve_hybrid_with_details(warm_query, top_k=10).hybrid
    reranker.rerank(warm_query, warm_hybrid, top_k=10)

    per_query = []
    latencies = {system: [] for system in SYSTEMS}
    for item in queries:
        query = item["query"]
        qrels = item["relevance"]

        dense, dense_timing = retrieve_with_timing(query, top_k=10)
        dense_seconds = (
            dense_timing.query_embedding_seconds + dense_timing.milvus_search_seconds
        )

        hybrid_result = retrieve_hybrid_with_details(query, top_k=10)
        hybrid_seconds = (
            hybrid_result.timing.query_embedding_seconds
            + hybrid_result.timing.milvus_search_seconds
            + hybrid_result.timing.bm25_seconds
            + hybrid_result.timing.fusion_and_filter_seconds
        )

        rerank_started = perf_counter()
        reranked = reranker.rerank(query, hybrid_result.hybrid, top_k=10)
        reranker_seconds = perf_counter() - rerank_started

        rankings = {
            "dense": dense,
            "hybrid": hybrid_result.hybrid,
            "hybrid_reranker": reranked,
        }
        metric_values = {
            system: metrics_for_ranking(
                [song.song_id for song in rankings[system]], qrels
            )
            for system in SYSTEMS
        }
        hybrid_ids = [song.song_id for song in hybrid_result.hybrid]
        reranked_ids = [song.song_id for song in reranked]
        shared = set(hybrid_ids) & set(reranked_ids)
        moved_up = [
            song_id
            for song_id in shared
            if reranked_ids.index(song_id) < hybrid_ids.index(song_id)
        ]
        moved_down = [
            song_id
            for song_id in shared
            if reranked_ids.index(song_id) > hybrid_ids.index(song_id)
        ]
        relevant = {song_id for song_id, grade in qrels.items() if grade > 0}
        recovered = list((set(reranked_ids[:5]) - set(hybrid_ids[:5])) & relevant)
        lost = list((set(hybrid_ids[:5]) - set(reranked_ids[:5])) & relevant)

        latencies["dense"].append(dense_seconds)
        latencies["hybrid"].append(hybrid_seconds)
        latencies["hybrid_reranker"].append(hybrid_seconds + reranker_seconds)
        per_query.append(
            {
                "query_id": item["query_id"],
                "query": query,
                "rationale": item["rationale"],
                "ground_truth": qrels,
                "constraints": {
                    "year_min": hybrid_result.constraints.year_min,
                    "year_max": hybrid_result.constraints.year_max,
                    "artist": hybrid_result.constraints.artist,
                    "genre_terms": list(hybrid_result.constraints.genre_terms),
                },
                "rankings": {
                    system: _ranking_payload(ranking)
                    for system, ranking in rankings.items()
                },
                "metrics": metric_values,
                "analysis": {
                    "moved_up": moved_up,
                    "moved_down": moved_down,
                    "recovered_relevant_at_5": recovered,
                    "lost_relevant_at_5": lost,
                    "filtered_by_metadata": hybrid_result.filtered_song_ids,
                },
                "latency_seconds": {
                    "dense": dense_seconds,
                    "hybrid": hybrid_seconds,
                    "reranker_only": reranker_seconds,
                    "hybrid_reranker_total": hybrid_seconds + reranker_seconds,
                },
            }
        )

    summary = {
        system: {
            **_mean_metrics(per_query, system),
            "mean_latency_seconds": statistics.fmean(latencies[system]),
        }
        for system in SYSTEMS
    }
    result = {
        "dataset_size": len(queries),
        "reranker_model": reranker.model_name,
        "cold_reranker_load_seconds": reranker_load_seconds,
        "summary": summary,
        "queries": per_query,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    result = evaluate()
    print(f"Evaluation queries: {result['dataset_size']}")
    print(f"Reranker: {result['reranker_model']}")
    print(f"Cold reranker load: {result['cold_reranker_load_seconds']:.6f}s")
    print("\nSystem                 Recall@5 Recall@10 MRR@10 nDCG@5 nDCG@10 P@5 Latency")
    for system in SYSTEMS:
        values = result["summary"][system]
        print(
            f"{system:<22} "
            f"{values['recall_at_5']:.4f}    {values['recall_at_10']:.4f}     "
            f"{values['mrr_at_10']:.4f}  {values['ndcg_at_5']:.4f}  "
            f"{values['ndcg_at_10']:.4f}   {values['precision_at_5']:.4f} "
            f"{values['mean_latency_seconds']:.6f}s"
        )
    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
