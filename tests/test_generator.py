from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag.generator import (  # noqa: E402
    GenerationUnavailableError,
    GroundingValidationError,
    RecommendationItem,
    RecommendationResponse,
    build_candidate_context,
    generate_recommendations,
    validate_recommendations,
)
from rockrag.retriever import RetrievedSong  # noqa: E402


def candidate(song_id: str = "candidate-a") -> RetrievedSong:
    return RetrievedSong(
        song_id=song_id,
        title="Catalog Title",
        artist="Catalog Artist",
        album="Catalog Album",
        release_year=1986,
        genres=["thrash metal"],
        tags=["fast", "1980s"],
        score=0.75,
        document_text="Title: Catalog Title",
    )


class FakeModels:
    def __init__(self, parsed: RecommendationResponse) -> None:
        self.parsed = parsed
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(parsed=self.parsed)


class FakeClient:
    def __init__(self, parsed: RecommendationResponse) -> None:
        self.models = FakeModels(parsed)


class FakeOllamaClient:
    def __init__(self, parsed: RecommendationResponse) -> None:
        self.content = parsed.model_dump_json()
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=self.content))


class UnavailableOllamaClient:
    def chat(self, **kwargs: object) -> None:
        raise ConnectionError("connection refused")


class GeneratorTests(unittest.TestCase):
    def test_candidate_context_contains_only_retrieved_fields(self) -> None:
        context = build_candidate_context([candidate()])
        self.assertIn("Candidate ID: candidate-a", context)
        self.assertIn("Dense cosine similarity: 0.750000", context)
        self.assertIn("Retrieval method: dense_cosine", context)
        self.assertIn("Genres: thrash metal", context)
        self.assertNotIn("confidence", context.lower())

    def test_invalid_song_id_is_rejected(self) -> None:
        response = RecommendationResponse(
            recommendations=[
                RecommendationItem(
                    song_id="metallica-nonexistent-song",
                    reason="Not a retrieved candidate.",
                )
            ]
        )
        with self.assertRaisesRegex(
            GroundingValidationError, "outside retrieved candidates"
        ):
            validate_recommendations(response, [candidate()], 5)

    def test_duplicate_song_id_is_rejected(self) -> None:
        response = RecommendationResponse(
            recommendations=[
                RecommendationItem(song_id="candidate-a", reason="First reason"),
                RecommendationItem(song_id="candidate-a", reason="Second reason"),
            ]
        )
        with self.assertRaisesRegex(GroundingValidationError, "duplicate song_id"):
            validate_recommendations(response, [candidate()], 5)

    def test_facts_are_backfilled_from_candidate(self) -> None:
        response = RecommendationResponse(
            recommendations=[
                RecommendationItem(
                    song_id="candidate-a", reason="Matches visible thrash tag."
                )
            ]
        )
        result = validate_recommendations(response, [candidate()], 5)[0]
        self.assertEqual(result.title, "Catalog Title")
        self.assertEqual(result.artist, "Catalog Artist")
        self.assertEqual(result.release_year, 1986)
        self.assertEqual(result.genres, ["thrash metal"])

    def test_missing_api_key_has_clear_generation_error(self) -> None:
        with patch("rockrag.generator.GEMINI_API_KEY", None):
            with self.assertRaisesRegex(
                GenerationUnavailableError, "GEMINI_API_KEY is not configured"
            ):
                generate_recommendations(
                    "thrash metal", [candidate()], provider="gemini"
                )

    def test_fake_gemini_structured_output_and_program_validation(self) -> None:
        parsed = RecommendationResponse(
            recommendations=[
                RecommendationItem(
                    song_id="candidate-a",
                    reason="The visible genre is thrash metal.",
                )
            ]
        )
        client = FakeClient(parsed)
        results = generate_recommendations(
            "fast thrash metal",
            [candidate()],
            client=client,
            model="test-model",
            provider="gemini",
        )

        self.assertEqual([item.song_id for item in results], ["candidate-a"])
        call = client.models.calls[0]
        self.assertEqual(call["model"], "test-model")
        config = call["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertIs(config.response_schema, RecommendationResponse)
        self.assertEqual(config.thinking_config.thinking_level.value, "LOW")
        self.assertIsNone(config.temperature)
        self.assertIn("Candidate ID: candidate-a", str(call["contents"]))

    def test_fake_ollama_structured_output_and_factual_backfill(self) -> None:
        parsed = RecommendationResponse(
            recommendations=[
                RecommendationItem(
                    song_id="candidate-a",
                    reason="The candidate has a visible thrash metal genre.",
                )
            ]
        )
        client = FakeOllamaClient(parsed)
        result = generate_recommendations(
            "fast thrash metal",
            [candidate()],
            client=client,
            model="test-qwen",
            provider="ollama",
        )[0]

        self.assertEqual(result.title, "Catalog Title")
        self.assertEqual(result.artist, "Catalog Artist")
        call = client.calls[0]
        self.assertEqual(call["model"], "test-qwen")
        self.assertEqual(call["format"], RecommendationResponse.model_json_schema())
        self.assertEqual(call["options"], {"temperature": 0})
        self.assertIn("Candidate ID: candidate-a", str(call["messages"]))

    def test_fake_ollama_invalid_candidate_is_rejected(self) -> None:
        client = FakeOllamaClient(
            RecommendationResponse(
                recommendations=[
                    RecommendationItem(song_id="not-retrieved", reason="Invalid")
                ]
            )
        )
        with self.assertRaisesRegex(
            GroundingValidationError, "outside retrieved candidates"
        ):
            generate_recommendations(
                "thrash metal", [candidate()], client=client, provider="ollama"
            )

    def test_fake_ollama_duplicate_candidate_is_rejected(self) -> None:
        client = FakeOllamaClient(
            RecommendationResponse(
                recommendations=[
                    RecommendationItem(song_id="candidate-a", reason="First"),
                    RecommendationItem(song_id="candidate-a", reason="Second"),
                ]
            )
        )
        with self.assertRaisesRegex(GroundingValidationError, "duplicate song_id"):
            generate_recommendations(
                "thrash metal", [candidate()], client=client, provider="ollama"
            )

    def test_ollama_unavailable_has_clear_error(self) -> None:
        with self.assertRaisesRegex(
            GenerationUnavailableError, "Ollama generation unavailable"
        ):
            generate_recommendations(
                "thrash metal",
                [candidate()],
                client=UnavailableOllamaClient(),
                provider="ollama",
            )


if __name__ == "__main__":
    unittest.main()
