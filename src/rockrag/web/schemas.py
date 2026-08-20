"""Explicit HTTP request and response schemas for the RockRAG web API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorResponse(StrictModel):
    error: str
    detail: str


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    catalog_size: int
    llm_provider: str
    llm_model: str
    ollama_reachable: bool


class SongResult(StrictModel):
    song_id: str
    title: str
    artist: str
    album: str | None = None
    release_year: int | None = None
    genres: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    reranker_score: float | None = None


class SearchRequest(StrictModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class SearchResponse(StrictModel):
    query: str
    results: list[SongResult]
    latency_seconds: float


class SongLookupResponse(StrictModel):
    status: Literal["found"] = "found"
    song_id: str
    title: str
    artist: str
    album: str | None = None
    release_year: int | None = None
    genres: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class CompareRequest(StrictModel):
    song_ids: list[str] = Field(min_length=2, max_length=5)


class SharedMetadata(StrictModel):
    shared_genres: list[str]
    shared_tags: list[str]


class SongDifference(StrictModel):
    song_id: str
    release_year: int | None = None
    unique_genres: list[str]
    unique_tags: list[str]


class CompareResponse(StrictModel):
    songs: list[SongResult]
    missing_song_ids: list[str]
    shared: SharedMetadata
    differences: list[SongDifference]


class AgentRequest(StrictModel):
    message: str = Field(min_length=1, max_length=2000)


class AgentToolCallResponse(StrictModel):
    step: int
    tool: str
    arguments: dict[str, Any]


class AgentToolResultResponse(StrictModel):
    step: int
    tool: str
    arguments: dict[str, Any]
    result: Any


class AgentResponse(StrictModel):
    answer: str
    steps: int
    latency_seconds: float
    tool_calls: list[AgentToolCallResponse]
    tool_results: list[AgentToolResultResponse]
