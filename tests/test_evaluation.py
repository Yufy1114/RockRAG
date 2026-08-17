from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.evaluate_retrieval import load_queries  # noqa: E402
from evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402


class EvaluationTests(unittest.TestCase):
    def test_qrels_schema_and_catalog_ids(self) -> None:
        queries = load_queries()
        self.assertGreaterEqual(len(queries), 12)
        self.assertEqual(len({item["query_id"] for item in queries}), len(queries))
        self.assertTrue(
            all(grade in (1, 2) for item in queries for grade in item["relevance"].values())
        )

    def test_recall_at_k(self) -> None:
        qrels = {"a": 2, "b": 1, "c": 1}
        self.assertAlmostEqual(recall_at_k(["a", "x", "b"], qrels, 2), 1 / 3)
        self.assertAlmostEqual(recall_at_k(["a", "x", "b"], qrels, 3), 2 / 3)

    def test_mrr_at_k(self) -> None:
        self.assertEqual(mrr_at_k(["x", "a", "b"], {"a": 2}, 10), 0.5)
        self.assertEqual(mrr_at_k(["x"], {"a": 2}, 1), 0.0)

    def test_ndcg_at_k(self) -> None:
        qrels = {"a": 2, "b": 1}
        self.assertEqual(ndcg_at_k(["a", "b"], qrels, 2), 1.0)
        self.assertLess(ndcg_at_k(["b", "a"], qrels, 2), 1.0)
        self.assertTrue(math.isfinite(ndcg_at_k(["x"], qrels, 5)))


if __name__ == "__main__":
    unittest.main()
