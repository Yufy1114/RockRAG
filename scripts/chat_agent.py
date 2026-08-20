"""Interactive CLI for the local RockRAG Ollama agent."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag.agent import AgentError, RockRAGAgent  # noqa: E402
from rockrag.config import OLLAMA_MODEL  # noqa: E402
from rockrag.tools import AVAILABLE_TOOLS  # noqa: E402


def print_trace(result) -> None:
    for call, tool_result in zip(result.tool_calls, result.tool_results, strict=True):
        print(f"\nStep {call.step}")
        print(f"LLM requested tool: {call.name}")
        print(f"Arguments: {call.arguments}")
        print(f"Tool result: {tool_result.serialized_result}")
    print(f"\nFinal Answer:\n{result.final_answer}")
    print(f"\nTotal steps: {result.steps}")
    print(f"Total latency: {result.latency_seconds:.3f} seconds")


def main() -> None:
    print("RockRAG Agent")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Tools: {', '.join(AVAILABLE_TOOLS)}")
    print("Type /help for help or /exit to quit.")
    agent = RockRAGAgent()
    while True:
        try:
            user_message = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if user_message == "/exit":
            return
        if user_message == "/help":
            print("Ask for catalog search, playlists, song metadata, or comparisons.")
            continue
        if not user_message:
            continue
        try:
            print_trace(agent.run(user_message))
        except AgentError as exc:
            print(f"Agent error: {exc}")


if __name__ == "__main__":
    main()
