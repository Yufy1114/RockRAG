from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag.agent import (  # noqa: E402
    AgentRunResult,
    AgentToolCall,
    AgentToolResult,
)
from rockrag.api import app, get_agent  # noqa: E402


SONG = {
    "song_id": "scorpions-still-loving-you",
    "title": "Still Loving You",
    "artist": "Scorpions",
    "album": "Love at First Sting",
    "release_year": 1984,
    "genres": ["hard rock"],
    "tags": ["power ballad", "melodic", "1980s"],
    "reranker_score": 1.25,
}


class FakeAgent:
    def run(self, message: str) -> AgentRunResult:
        call = AgentToolCall(1, "search_songs", {"query": message, "top_k": 5})
        result = AgentToolResult(
            1,
            "search_songs",
            call.arguments,
            [SONG],
            "[]",
        )
        return AgentRunResult("A grounded answer.", [call], [result], 2, 0.25)


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    @patch("rockrag.api.ollama_is_reachable", return_value=True)
    def test_health(self, _reachable) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["catalog_size"], 38)
        self.assertIn("llm_model", body)

    @patch("rockrag.api.search_songs", return_value=[SONG])
    def test_search(self, search_mock) -> None:
        response = self.client.post("/api/search", json={"query": "power ballad", "top_k": 5})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"][0]["song_id"], SONG["song_id"])
        search_mock.assert_called_once_with("power ballad", 5)

    @patch("rockrag.api.search_songs", side_effect=RuntimeError("index missing"))
    def test_search_index_unavailable(self, _search_mock) -> None:
        response = self.client.post("/api/search", json={"query": "power ballad"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"], "service_unavailable")

    def test_valid_song(self) -> None:
        response = self.client.get("/api/song/scorpions-still-loving-you")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Still Loving You")

    def test_missing_song(self) -> None:
        response = self.client.get("/api/song/not-in-catalog")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"], "song_not_found")

    def test_compare(self) -> None:
        response = self.client.post(
            "/api/compare",
            json={"song_ids": ["scorpions-still-loving-you", "whitesnake-is-this-love"]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["songs"]), 2)

    @patch("rockrag.api.ollama_is_reachable", return_value=True)
    def test_agent_with_mock(self, _reachable) -> None:
        app.dependency_overrides[get_agent] = lambda: FakeAgent()
        response = self.client.post("/api/agent", json={"message": "recommend"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "A grounded answer.")
        self.assertEqual(body["tool_calls"][0]["tool"], "search_songs")
        self.assertEqual(body["tool_results"][0]["result"][0]["title"], "Still Loving You")

    def test_invalid_request_schema(self) -> None:
        response = self.client.post("/api/search", json={"query": "", "top_k": 99})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "invalid_request")

    @patch("rockrag.api.ollama_is_reachable", return_value=False)
    def test_ollama_unavailable(self, _reachable) -> None:
        app.dependency_overrides[get_agent] = lambda: FakeAgent()
        response = self.client.post("/api/agent", json={"message": "recommend"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"], "ollama_unavailable")

    def test_static_home(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("RockRAG", response.text)


if __name__ == "__main__":
    unittest.main()
