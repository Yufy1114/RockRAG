from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag.embeddings import (  # noqa: E402
    EMBEDDING_DIMENSION,
    QUERY_INSTRUCTION,
    embed_documents,
    embed_query,
)


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def encode(self, value: object, **kwargs: object) -> np.ndarray:
        self.calls.append((value, kwargs))
        if isinstance(value, list):
            output = np.zeros((len(value), EMBEDDING_DIMENSION), dtype=np.float32)
            output[:, 0] = 1.0
            return output
        output = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        output[0] = 1.0
        return output


class EmbeddingTests(unittest.TestCase):
    def test_document_shape_normalization_and_no_prefix(self) -> None:
        model = FakeModel()
        with patch("rockrag.embeddings.load_model", return_value=model):
            embeddings = embed_documents(["document one", "document two"])

        self.assertEqual(embeddings.shape, (2, EMBEDDING_DIMENSION))
        self.assertTrue(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0))
        self.assertFalse(np.isnan(embeddings).any())
        self.assertFalse(np.isinf(embeddings).any())
        self.assertEqual(model.calls[0][0], ["document one", "document two"])
        self.assertIs(model.calls[0][1]["normalize_embeddings"], True)

    def test_query_shape_instruction_and_normalization(self) -> None:
        model = FakeModel()
        with patch("rockrag.embeddings.load_model", return_value=model):
            embedding = embed_query("heavy metal")

        self.assertEqual(embedding.shape, (EMBEDDING_DIMENSION,))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0)
        self.assertEqual(model.calls[0][0], QUERY_INSTRUCTION + "heavy metal")
        self.assertIs(model.calls[0][1]["normalize_embeddings"], True)

    def test_empty_document_batch_has_matrix_shape(self) -> None:
        embeddings = embed_documents([])
        self.assertEqual(embeddings.shape, (0, EMBEDDING_DIMENSION))


if __name__ == "__main__":
    unittest.main()
