"""Small transparent retrieval metrics for graded RockRAG qrels."""

from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    """Fraction of all grade>0 relevant songs retrieved in the first K."""

    relevant = {song_id for song_id, grade in qrels.items() if grade > 0}
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def precision_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    """Fraction of the first K positions occupied by grade>0 songs."""

    if k <= 0:
        raise ValueError("k must be positive.")
    relevant_count = sum(qrels.get(song_id, 0) > 0 for song_id in ranked_ids[:k])
    return relevant_count / k


def mrr_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    """Reciprocal rank of the first grade>0 result within K."""

    for rank, song_id in enumerate(ranked_ids[:k], start=1):
        if qrels.get(song_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def _dcg(grades: list[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def ndcg_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    """Normalized graded gain, rewarding high grades near the top."""

    actual = [qrels.get(song_id, 0) for song_id in ranked_ids[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    return _dcg(actual) / ideal_dcg if ideal_dcg else 0.0


def metrics_for_ranking(
    ranked_ids: list[str], qrels: dict[str, int]
) -> dict[str, float]:
    return {
        "recall_at_5": recall_at_k(ranked_ids, qrels, 5),
        "recall_at_10": recall_at_k(ranked_ids, qrels, 10),
        "precision_at_5": precision_at_k(ranked_ids, qrels, 5),
        "mrr_at_10": mrr_at_k(ranked_ids, qrels, 10),
        "ndcg_at_5": ndcg_at_k(ranked_ids, qrels, 5),
        "ndcg_at_10": ndcg_at_k(ranked_ids, qrels, 10),
    }
