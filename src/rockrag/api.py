"""Thin FastAPI adapter over the existing RockRAG retrieval and agent layers."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Annotated
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ollama import RequestError as OllamaRequestError, ResponseError as OllamaResponseError

from .agent import (
    AgentError,
    MaxAgentStepsError,
    RockRAGAgent,
    ToolArgumentsError,
    UnknownToolError,
)
from .catalog_loader import load_catalog
from .config import GEMINI_MODEL, LLM_PROVIDER, OLLAMA_HOST, OLLAMA_MODEL, PROJECT_ROOT
from .tools import compare_songs, get_song, search_songs
from .web.schemas import (
    AgentRequest,
    AgentResponse,
    AgentToolCallResponse,
    AgentToolResultResponse,
    CompareRequest,
    CompareResponse,
    ErrorResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SongLookupResponse,
)


WEB_ROOT = PROJECT_ROOT / "web"
ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}

app = FastAPI(
    title="RockRAG API",
    version="0.8.0",
    description="Hybrid music retrieval and an auditable local tool-calling agent.",
)


def get_agent() -> RockRAGAgent:
    """Dependency boundary used by production and mocked API tests."""

    return RockRAGAgent()


def _error(status_code: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


@app.exception_handler(RequestValidationError)
async def request_validation_error(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _error(400, "invalid_request", str(exc))


@app.exception_handler(UnknownToolError)
@app.exception_handler(ToolArgumentsError)
async def unsafe_tool_error(_request: Request, exc: AgentError) -> JSONResponse:
    return _error(400, "agent_tool_error", str(exc))


@app.exception_handler(MaxAgentStepsError)
async def max_steps_error(_request: Request, exc: MaxAgentStepsError) -> JSONResponse:
    return _error(500, "agent_step_limit", str(exc))


@app.exception_handler(AgentError)
async def agent_error(_request: Request, exc: AgentError) -> JSONResponse:
    return _error(500, "agent_error", str(exc))


@app.exception_handler(HTTPException)
async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
    error = {
        404: "not_found",
        503: "service_unavailable",
    }.get(exc.status_code, "invalid_request")
    return _error(exc.status_code, error, str(exc.detail))


@app.exception_handler(Exception)
async def unexpected_error(_request: Request, _exc: Exception) -> JSONResponse:
    return _error(500, "internal_error", "An unexpected internal error occurred.")


def ollama_is_reachable(timeout_seconds: float = 1.0) -> bool:
    """Probe Ollama's lightweight tags endpoint without running generation."""

    endpoint = f"{OLLAMA_HOST.rstrip('/')}/api/tags"
    try:
        with urlopen(endpoint, timeout=timeout_seconds) as response:  # noqa: S310
            return 200 <= response.status < 300
    except (HTTPError, URLError, OSError, TimeoutError):
        return False


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/health", response_model=HealthResponse, responses=ERROR_RESPONSES)
def health() -> HealthResponse:
    model = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else GEMINI_MODEL
    return HealthResponse(
        catalog_size=len(load_catalog()),
        llm_provider=LLM_PROVIDER,
        llm_model=model,
        ollama_reachable=ollama_is_reachable(),
    )


@app.post("/api/agent", response_model=AgentResponse, responses=ERROR_RESPONSES)
def run_agent(
    request: AgentRequest,
    agent: Annotated[RockRAGAgent, Depends(get_agent)],
) -> AgentResponse | JSONResponse:
    if LLM_PROVIDER == "ollama" and not ollama_is_reachable():
        return _error(503, "ollama_unavailable", "Local LLM unavailable at configured OLLAMA_HOST.")
    try:
        result = agent.run(request.message)
    except (
        ConnectionError,
        OSError,
        TimeoutError,
        OllamaRequestError,
        OllamaResponseError,
    ) as exc:
        return _error(503, "ollama_unavailable", f"Local LLM request failed: {exc}")
    return AgentResponse(
        answer=result.final_answer,
        steps=result.steps,
        latency_seconds=result.latency_seconds,
        tool_calls=[
            AgentToolCallResponse(step=item.step, tool=item.name, arguments=item.arguments)
            for item in result.tool_calls
        ],
        tool_results=[
            AgentToolResultResponse(
                step=item.step,
                tool=item.name,
                arguments=item.arguments,
                result=item.result,
            )
            for item in result.tool_results
        ],
    )


@app.post("/api/search", response_model=SearchResponse, responses=ERROR_RESPONSES)
def search(request: SearchRequest) -> SearchResponse:
    started = perf_counter()
    try:
        results = search_songs(request.query, request.top_k)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Retrieval service or Milvus index unavailable: {exc}",
        ) from exc
    return SearchResponse(
        query=request.query,
        results=results,
        latency_seconds=perf_counter() - started,
    )


@app.get(
    "/api/song/{song_id}",
    response_model=SongLookupResponse,
    responses=ERROR_RESPONSES,
)
def song(song_id: str) -> SongLookupResponse | JSONResponse:
    result = get_song(song_id)
    if result["status"] == "not_found":
        return _error(404, "song_not_found", f"No catalog song has ID {song_id!r}.")
    return SongLookupResponse.model_validate(result)


@app.post("/api/compare", response_model=CompareResponse, responses=ERROR_RESPONSES)
def compare(request: CompareRequest) -> CompareResponse | JSONResponse:
    try:
        result = compare_songs(request.song_ids)
    except ValueError as exc:
        return _error(400, "invalid_tool_arguments", str(exc))
    return CompareResponse.model_validate(result)


app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")
