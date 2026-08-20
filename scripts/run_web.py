"""Run the RockRAG FastAPI application without package installation."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag.config import WEB_HOST, WEB_PORT  # noqa: E402


if __name__ == "__main__":
    uvicorn.run("rockrag.api:app", host=WEB_HOST, port=WEB_PORT)
