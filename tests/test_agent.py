from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag.agent import (  # noqa: E402
    MaxAgentStepsError,
    RockRAGAgent,
    ToolArgumentsError,
    UnknownToolError,
    dispatch_tool,
)


def tool_call(name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def response(*, calls: list | None = None, content: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(tool_calls=calls or [], content=content)
    )


class FakeClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class AgentTests(unittest.TestCase):
    def test_valid_dispatch_and_serialization(self) -> None:
        registry = {"echo": lambda value: {"value": value}}
        validated, result, serialized = dispatch_tool(
            "echo", {"value": "x"}, registry=registry
        )
        self.assertEqual(validated, {"value": "x"})
        self.assertEqual(result, {"value": "x"})
        self.assertEqual(json.loads(serialized), result)

    def test_unknown_tool_rejected(self) -> None:
        with self.assertRaises(UnknownToolError):
            dispatch_tool("shell", {}, registry={})

    def test_malformed_arguments_rejected(self) -> None:
        def count_tool(count: int) -> dict:
            return {"count": count}

        with self.assertRaises(ToolArgumentsError):
            dispatch_tool("count_tool", {"wrong": 2}, registry={"count_tool": count_tool})

    def test_multi_tool_conversation_and_trace(self) -> None:
        def first(value: str) -> dict:
            return {"first": value}

        def second(count: int) -> list[int]:
            return list(range(count))

        client = FakeClient([
            response(calls=[tool_call("first", {"value": "a"})]),
            response(calls=[tool_call("second", {"count": 2})]),
            response(content="Grounded final answer"),
        ])
        result = RockRAGAgent(
            client=client,
            registry={"first": first, "second": second},
            max_steps=5,
        ).run("Do two things")
        self.assertEqual([call.name for call in result.tool_calls], ["first", "second"])
        self.assertEqual(result.final_answer, "Grounded final answer")
        self.assertEqual(result.steps, 3)
        tool_messages = [
            message
            for call in client.calls
            for message in call["messages"]
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        self.assertTrue(any(json.loads(message["content"]) == {"first": "a"} for message in tool_messages))

    def test_max_agent_steps(self) -> None:
        def repeat() -> dict:
            return {"ok": True}

        client = FakeClient([
            response(calls=[tool_call("repeat", {})]),
            response(calls=[tool_call("repeat", {})]),
        ])
        with self.assertRaises(MaxAgentStepsError):
            RockRAGAgent(client=client, registry={"repeat": repeat}, max_steps=2).run("loop")

    def test_unknown_model_tool_call_is_rejected(self) -> None:
        client = FakeClient([response(calls=[tool_call("web_search", {"q": "x"})])])
        with self.assertRaises(UnknownToolError):
            RockRAGAgent(client=client, registry={}, max_steps=1).run("search web")


if __name__ == "__main__":
    unittest.main()
