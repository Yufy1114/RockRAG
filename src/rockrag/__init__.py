"""RockRAG core package."""

from .catalog_loader import CatalogValidationError, load_catalog
from .agent import AgentRunResult, RockRAGAgent
from .embeddings import embed_documents, embed_query
from .generator import Recommendation, generate_recommendations
from .hybrid_retriever import HybridRetrievedSong, QueryConstraints, retrieve_hybrid
from .models import SongRecord
from .reranker import RerankedSong, rerank
from .retriever import RetrievedSong, retrieve
from .song_document import song_to_document

__all__ = [
    "CatalogValidationError",
    "AgentRunResult",
    "RockRAGAgent",
    "SongRecord",
    "RetrievedSong",
    "Recommendation",
    "HybridRetrievedSong",
    "QueryConstraints",
    "RerankedSong",
    "embed_documents",
    "embed_query",
    "generate_recommendations",
    "load_catalog",
    "retrieve",
    "retrieve_hybrid",
    "rerank",
    "song_to_document",
]
