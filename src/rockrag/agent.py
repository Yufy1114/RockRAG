"""Minimal auditable Ollama tool-calling loop for RockRAG."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping, get_type_hints

from ollama import Client as OllamaClient
from pydantic import TypeAdapter, ValidationError

from .config import MAX_AGENT_STEPS, OLLAMA_HOST, OLLAMA_MODEL
from .tools import AVAILABLE_TOOLS


AGENT_SYSTEM_PROMPT = """You are RockRAG, a music discovery agent focused on hard rock and metal.
Use tools when factual catalog information or recommendations are required.
Never invent songs or metadata. Only make factual claims supported by tool results.
If the catalog does not contain enough matching songs, say so.
For recommendation requests, prefer retrieval tools rather than model memory.
Do not mention songs outside tool results when presenting catalog-based recommendations.
Never guess or invent song IDs. When the user gives song titles instead of exact
catalog IDs, first call search_songs to resolve the real IDs before using
get_song or compare_songs.
Use compare_songs for factual comparisons. For a request that combines comparison and discovery, call the comparison tool and a retrieval tool before answering.
"""


class AgentError(RuntimeError):
    """Base error for a failed or unsafe agent run."""


class UnknownToolError(AgentError):
    """Raised when the model requests a tool outside the explicit registry."""


class ToolArgumentsError(AgentError):
    """Raised when a tool call does not match the Python function signature."""


class MaxAgentStepsError(AgentError):
    """Raised when tool calling does not terminate within the configured limit."""


@dataclass(frozen=True, slots=True)
class AgentToolCall:
    step: int
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentToolResult:
    step: int
    name: str
    arguments: dict[str, Any]
    result: Any
    serialized_result: str


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    final_answer: str
    tool_calls: list[AgentToolCall]
    tool_results: list[AgentToolResult]
    steps: int
    latency_seconds: float


def _validate_tool_arguments(
    function: Callable[..., Any], arguments: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(arguments, Mapping):
        raise ToolArgumentsError("Tool arguments must be a JSON object.")
    signature = inspect.signature(function)
    hints = get_type_hints(function)
    unknown = set(arguments) - set(signature.parameters)
    if unknown:
        raise ToolArgumentsError(f"Unknown arguments for {function.__name__}: {sorted(unknown)}")

    validated: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name not in arguments:
            if parameter.default is inspect.Parameter.empty:
                raise ToolArgumentsError(
                    f"Missing required argument for {function.__name__}: {name}"
                )
            validated[name] = parameter.default
            continue
        annotation = hints.get(name, Any)
        try:
            validated[name] = TypeAdapter(annotation).validate_python(arguments[name])
        except ValidationError as exc:
            raise ToolArgumentsError(
                f"Invalid argument {name!r} for {function.__name__}: {exc}"
            ) from exc
    return validated


def dispatch_tool(
    name: str,
    arguments: Mapping[str, Any],
    *,
    registry: Mapping[str, Callable[..., Any]] = AVAILABLE_TOOLS,
) -> tuple[dict[str, Any], Any, str]:
    """Validate, execute, and JSON-serialize one explicitly registered tool."""

    function = registry.get(name)
    if function is None:
        raise UnknownToolError(f"Unknown tool requested: {name!r}.")
    validated = _validate_tool_arguments(function, arguments)
    try:
        result = function(**validated)
    except (TypeError, ValueError) as exc:
        raise ToolArgumentsError(f"Tool {name!r} rejected its arguments: {exc}") from exc
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True)
    return validated, result, serialized


class RockRAGAgent:
    """Run Ollama native tool calling with a strict step and tool boundary."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
        registry: Mapping[str, Callable[..., Any]] = AVAILABLE_TOOLS,
        max_steps: int = MAX_AGENT_STEPS,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self.client = client or OllamaClient(host=host)
        self.model = model
        self.registry = dict(registry)
        self.max_steps = max_steps

    def run(self, user_message: str) -> AgentRunResult:
        """Execute tool calls until Qwen returns a final answer or hits the cap."""

        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message must be a non-empty string.")
        started = perf_counter()
        messages: list[Any] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message.strip()},
        ]
        calls: list[AgentToolCall] = []
        results: list[AgentToolResult] = []
        tool_schemas = list(self.registry.values())

        for step in range(1, self.max_steps + 1):
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=tool_schemas,
                think=False,
                options={"temperature": 0},
            )
            message = response.message
            tool_calls = list(message.tool_calls or [])
            messages.append(message)
            if not tool_calls:
                final_answer = (message.content or "").strip()
                if not final_answer:
                    raise AgentError("Agent returned neither tool calls nor a final answer.")
                return AgentRunResult(
                    final_answer=final_answer,
                    tool_calls=calls,
                    tool_results=results,
                    steps=step,
                    latency_seconds=perf_counter() - started,
                )

            for tool_call in tool_calls:
                name = tool_call.function.name
                raw_arguments = dict(tool_call.function.arguments)
                validated, result, serialized = dispatch_tool(
                    name, raw_arguments, registry=self.registry
                )
                calls.append(AgentToolCall(step, name, validated))
                results.append(AgentToolResult(step, name, validated, result, serialized))
                messages.append(
                    {"role": "tool", "tool_name": name, "content": serialized}
                )

        raise MaxAgentStepsError(
            f"Agent exceeded MAX_AGENT_STEPS={self.max_steps} without a final answer."
        )
