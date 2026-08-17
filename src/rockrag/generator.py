"""Grounded recommendation generation from already retrieved songs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types
from ollama import Client as OllamaClient
from pydantic import BaseModel, Field
from pydantic import ValidationError

from .config import (
    DEFAULT_RECOMMENDATION_COUNT,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OLLAMA_HOST,
    OLLAMA_MODEL,
)
from .retriever import RetrievedSong


SYSTEM_INSTRUCTIONS = """You are the recommendation generation layer of RockRAG.

You are not a retriever. Select songs only from the retrieved candidates supplied by the application.

Rules:
1. Recommend only song_id values present in the candidate list.
2. Never add a song from memory or outside the candidate list.
3. Do not supplement candidate facts from your own knowledge.
4. Base each reason only on visible genres, tags, release year, artist, album, and retrieval relevance.
5. If candidate metadata does not support an attribute, do not claim that attribute.
6. Call the retrieval value a similarity score, never confidence, probability, or accuracy.
7. If the candidates are a weak match, state that the available candidates are limited instead of inventing a better song.
8. Do not return duplicate song IDs.
9. Return no more than the requested recommendation count.
10. Output only song_id and reason for each recommendation. The application supplies all factual display fields.
"""


class RecommendationItem(BaseModel):
    """The only fields the model is allowed to generate for one selection."""

    song_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class RecommendationResponse(BaseModel):
    recommendations: list[RecommendationItem]


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Validated recommendation with facts backfilled from RetrievedSong."""

    song_id: str
    title: str
    artist: str
    album: str | None
    release_year: int | None
    genres: list[str]
    tags: list[str]
    retrieval_score: float
    reason: str


class GenerationUnavailableError(RuntimeError):
    """Raised when generation cannot run because configuration is missing."""


class GroundingValidationError(ValueError):
    """Raised when structured model output violates candidate grounding."""


def _display_list(values: list[str]) -> str:
    return ", ".join(values) if values else "Not available"


def build_candidate_context(candidates: list[RetrievedSong]) -> str:
    """Expose retrieved facts with a score label matching the retrieval stage."""

    blocks = []
    for candidate in candidates:
        if hasattr(candidate, "reranker_score"):
            score_lines = [
                "Retrieval method: cross_encoder_reranker",
                f"Reranker relevance score: {candidate.reranker_score:.6f}",
            ]
        elif hasattr(candidate, "fusion_score"):
            score_lines = [
                "Retrieval method: hybrid_rrf",
                f"RRF fusion score: {candidate.fusion_score:.6f}",
            ]
        else:
            score_lines = [
                "Retrieval method: dense_cosine",
                f"Dense cosine similarity: {candidate.score:.6f}",
            ]
        blocks.append(
            "\n".join(
                [
                    f"Candidate ID: {candidate.song_id}",
                    f"Title: {candidate.title}",
                    f"Artist: {candidate.artist}",
                    f"Album: {candidate.album or 'Not available'}",
                    f"Release year: {candidate.release_year if candidate.release_year is not None else 'Not available'}",
                    f"Genres: {_display_list(candidate.genres)}",
                    f"Tags: {_display_list(candidate.tags)}",
                    *score_lines,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def validate_recommendations(
    response: RecommendationResponse,
    candidates: list[RetrievedSong],
    recommendation_count: int,
) -> list[Recommendation]:
    """Reject ungrounded/duplicate output and backfill every factual field."""

    if recommendation_count <= 0:
        raise ValueError("recommendation_count must be positive.")
    if len(response.recommendations) > recommendation_count:
        raise GroundingValidationError(
            "Model returned more recommendations than requested: "
            f"{len(response.recommendations)} > {recommendation_count}."
        )

    candidates_by_id = {candidate.song_id: candidate for candidate in candidates}
    seen_ids: set[str] = set()
    validated: list[Recommendation] = []

    for item in response.recommendations:
        song_id = item.song_id.strip()
        reason = item.reason.strip()
        if song_id not in candidates_by_id:
            raise GroundingValidationError(
                f"Model returned song_id outside retrieved candidates: {song_id!r}."
            )
        if song_id in seen_ids:
            raise GroundingValidationError(
                f"Model returned duplicate song_id: {song_id!r}."
            )
        if not reason:
            raise GroundingValidationError(
                f"Model returned an empty reason for song_id {song_id!r}."
            )

        candidate = candidates_by_id[song_id]
        seen_ids.add(song_id)
        validated.append(
            Recommendation(
                song_id=candidate.song_id,
                title=candidate.title,
                artist=candidate.artist,
                album=candidate.album,
                release_year=candidate.release_year,
                genres=list(candidate.genres),
                tags=list(candidate.tags),
                retrieval_score=candidate.score,
                reason=reason,
            )
        )

    return validated


def generate_recommendations(
    user_query: str,
    candidates: list[RetrievedSong],
    recommendation_count: int = DEFAULT_RECOMMENDATION_COUNT,
    *,
    api_key: str | None = None,
    model: str | None = None,
    client: Any | None = None,
    provider: str | None = None,
    host: str | None = None,
) -> list[Recommendation]:
    """Generate structured explanations, then enforce candidate grounding."""

    if not user_query.strip():
        raise ValueError("user_query must be a non-empty string.")
    if not candidates:
        return []
    if recommendation_count <= 0:
        raise ValueError("recommendation_count must be positive.")

    context = build_candidate_context(candidates)
    user_input = (
        f"User request:\n{user_query.strip()}\n\n"
        f"Select at most {recommendation_count} recommendations from these "
        f"{len(candidates)} retrieved candidates:\n\n{context}"
    )

    resolved_provider = (provider or LLM_PROVIDER).strip().lower()
    if resolved_provider == "gemini":
        resolved_api_key = api_key or GEMINI_API_KEY
        if client is None:
            if not resolved_api_key:
                raise GenerationUnavailableError(
                    "Gemini generation unavailable because GEMINI_API_KEY is not configured."
                )
            client = genai.Client(api_key=resolved_api_key)
        response = client.models.generate_content(
            model=model or GEMINI_MODEL,
            contents=user_input,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTIONS,
                response_mime_type="application/json",
                response_schema=RecommendationResponse,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
                max_output_tokens=2000,
            ),
        )
        parsed = response.parsed
        if parsed is None:
            raise GroundingValidationError(
                "Gemini response did not contain a parsed recommendation response."
            )
        if not isinstance(parsed, RecommendationResponse):
            parsed = RecommendationResponse.model_validate(parsed)
    elif resolved_provider == "ollama":
        if client is None:
            client = OllamaClient(host=host or OLLAMA_HOST)
        try:
            response = client.chat(
                model=model or OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_input},
                ],
                format=RecommendationResponse.model_json_schema(),
                options={"temperature": 0},
            )
        except Exception as exc:
            raise GenerationUnavailableError(
                f"Ollama generation unavailable at {host or OLLAMA_HOST}: {exc}"
            ) from exc
        try:
            parsed = RecommendationResponse.model_validate_json(
                response.message.content
            )
        except ValidationError as exc:
            raise GroundingValidationError(
                "Ollama response did not match RecommendationResponse."
            ) from exc
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER {resolved_provider!r}; use 'gemini' or 'ollama'."
        )
    return validate_recommendations(parsed, candidates, recommendation_count)
