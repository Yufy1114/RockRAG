"""Catalog-only tools exposed to the RockRAG Ollama agent."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .catalog_loader import load_catalog
from .hybrid_retriever import retrieve_hybrid
from .reranker import CrossEncoderReranker, RerankedSong, rerank


def _song_payload(song: Any) -> dict[str, Any]:
    payload = {
        "song_id": song.song_id,
        "title": song.title,
        "artist": song.artist,
        "album": song.album,
        "release_year": song.release_year,
        "genres": list(song.genres),
        "tags": list(song.tags),
    }
    if hasattr(song, "reranker_score"):
        payload["reranker_score"] = float(song.reranker_score)
    return payload


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


def _retrieve_and_rerank(query: str, count: int) -> list[RerankedSong]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 10:
        raise ValueError("count must be an integer from 1 to 10.")
    candidates = retrieve_hybrid(query.strip(), top_k=10)
    return rerank(query.strip(), candidates, top_k=count, reranker=_get_reranker())


def search_songs(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search the RockRAG catalog for hard-rock/metal songs.

    Use this for discovery or to find candidates related to a natural-language
    music preference. Results come from Hybrid retrieval followed by the
    CrossEncoder; the function never invents songs or writes natural-language
    recommendations.

    Args:
        query: Natural-language music search request.
        top_k: Number of candidates to return, from 1 through 10.

    Returns:
        Ranked catalog songs with factual metadata and reranker scores.
    """

    return [_song_payload(song) for song in _retrieve_and_rerank(query, top_k)]


def get_song(song_id: str) -> dict[str, Any]:
    """Get factual metadata for one exact RockRAG catalog song ID.

    Args:
        song_id: Stable catalog identifier, normally obtained from another tool.

    Returns:
        A found result with catalog metadata, or an explicit not_found result.
    """

    if not isinstance(song_id, str) or not song_id.strip():
        raise ValueError("song_id must be a non-empty string.")
    normalized_id = song_id.strip()
    for song in load_catalog():
        if song.song_id == normalized_id:
            return {"status": "found", **_song_payload(song)}
    return {"status": "not_found", "song_id": normalized_id}


def compare_songs(song_ids: list[str]) -> dict[str, Any]:
    """Compare factual metadata for two to five catalog songs.

    Use exact song IDs obtained from search results. Missing IDs are reported;
    no metadata is inferred or generated.

    Args:
        song_ids: Two to five stable RockRAG song IDs.

    Returns:
        Song metadata, missing IDs, shared genres/tags, and per-song differences.
    """

    if not isinstance(song_ids, list) or not 2 <= len(song_ids) <= 5:
        raise ValueError("song_ids must contain between 2 and 5 IDs.")
    if any(not isinstance(song_id, str) or not song_id.strip() for song_id in song_ids):
        raise ValueError("Every song_id must be a non-empty string.")
    normalized_ids = [song_id.strip() for song_id in song_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("song_ids must not contain duplicates.")

    catalog = {song.song_id: song for song in load_catalog()}
    found = [catalog[song_id] for song_id in normalized_ids if song_id in catalog]
    missing = [song_id for song_id in normalized_ids if song_id not in catalog]
    genre_sets = [set(song.genres) for song in found]
    tag_sets = [set(song.tags) for song in found]
    shared_genres = sorted(set.intersection(*genre_sets)) if genre_sets else []
    shared_tags = sorted(set.intersection(*tag_sets)) if tag_sets else []

    return {
        "songs": [_song_payload(song) for song in found],
        "missing_song_ids": missing,
        "shared": {
            "shared_genres": shared_genres,
            "shared_tags": shared_tags,
        },
        "differences": [
            {
                "song_id": song.song_id,
                "release_year": song.release_year,
                "unique_genres": sorted(set(song.genres) - set(shared_genres)),
                "unique_tags": sorted(set(song.tags) - set(shared_tags)),
            }
            for song in found
        ],
    }


def build_playlist(query: str, count: int = 5) -> list[dict[str, Any]]:
    """Build a final catalog-only playlist for a music request.

    This uses the existing Hybrid retrieval, hard metadata filtering, and
    CrossEncoder reranker. It does not call an LLM or generate explanations.

    Args:
        query: Natural-language playlist requirements.
        count: Requested playlist size, from 1 through 10.

    Returns:
        Up to count ranked catalog songs; fewer are returned when hard filters
        leave insufficient matching catalog coverage.
    """

    return [_song_payload(song) for song in _retrieve_and_rerank(query, count)]


AVAILABLE_TOOLS = {
    "search_songs": search_songs,
    "get_song": get_song,
    "compare_songs": compare_songs,
    "build_playlist": build_playlist,
}
