"""Configuration shared by the RockRAG pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "songs.json"
DEFAULT_MUSICBRAINZ_SNAPSHOT_PATH = (
    PROJECT_ROOT / "data" / "raw" / "musicbrainz_snapshot.json"
)
MODEL_CACHE_DIR = PROJECT_ROOT / "storage" / "models"

MILVUS_DATABASE_PATH = PROJECT_ROOT / "storage" / "rockrag.db"
MILVUS_COLLECTION_NAME = "songs"
MILVUS_EMBEDDING_DIMENSION = 384
MILVUS_METRIC_TYPE = "COSINE"
MILVUS_INDEX_TYPE = "FLAT"
DEFAULT_TOP_K = 5

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GENERATION_RETRIEVAL_TOP_K = 10
DEFAULT_RECOMMENDATION_COUNT = 5

HYBRID_RRF_CONSTANT = 60
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANK_TOP_K = 10
