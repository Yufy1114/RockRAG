"""Transparent BM25, RRF, and hard-metadata filtering for Phase 5."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from time import perf_counter

from rank_bm25 import BM25Okapi

from .catalog_loader import load_catalog
from .config import DEFAULT_CATALOG_PATH, HYBRID_RRF_CONSTANT
from .models import SongRecord
from .retriever import RetrievedSong, RetrievalTiming, retrieve_with_timing
from .song_document import song_to_document


TOKEN_PATTERN = re.compile(r"[\w']+")
DECADE_PATTERN = re.compile(r"(?<!\d)(?:19)?(80|90)s\b", re.IGNORECASE)
GENRE_PHRASES = (
    "traditional heavy metal",
    "progressive metal",
    "melodic metal",
    "thrash metal",
    "power metal",
    "heavy metal",
    "hard rock",
    "power ballad",
    "glam metal",
    "hair metal",
    "nwobhm",
)


@dataclass(frozen=True, slots=True)
class QueryConstraints:
    """Deterministic query constraints; genre terms remain soft preferences."""

    year_min: int | None = None
    year_max: int | None = None
    artist: str | None = None
    genre_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HybridRetrievedSong(RetrievedSong):
    """RetrievedSong plus auditable ranks and scores from both retrieval paths."""

    dense_rank: int | None
    dense_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    fusion_score: float


@dataclass(frozen=True, slots=True)
class HybridTiming:
    query_embedding_seconds: float
    milvus_search_seconds: float
    bm25_seconds: float
    fusion_and_filter_seconds: float


@dataclass(frozen=True, slots=True)
class HybridRetrievalResult:
    constraints: QueryConstraints
    dense: list[RetrievedSong]
    bm25: list[RetrievedSong]
    hybrid: list[HybridRetrievedSong]
    filtered_song_ids: list[str]
    timing: HybridTiming


def tokenize(text: str) -> list[str]:
    """Use one small, deterministic tokenizer for BM25 documents and queries."""

    return TOKEN_PATTERN.findall(text.lower())


def parse_query_constraints(
    query: str,
    *,
    known_artists: list[str] | None = None,
) -> QueryConstraints:
    """Extract only explicit decades/artists plus auditable soft genre phrases."""

    normalized = query.casefold()
    year_min = year_max = None
    decade_match = DECADE_PATTERN.search(normalized)
    if decade_match:
        decade_start = 1900 + int(decade_match.group(1))
        year_min, year_max = decade_start, decade_start + 9

    artist = None
    if known_artists:
        matches = [
            name
            for name in known_artists
            if re.search(rf"(?<!\w){re.escape(name.casefold())}(?!\w)", normalized)
        ]
        if matches:
            artist = max(matches, key=len)

    genre_terms = tuple(term for term in GENRE_PHRASES if term in normalized)
    return QueryConstraints(
        year_min=year_min,
        year_max=year_max,
        artist=artist,
        genre_terms=genre_terms,
    )


def _song_to_retrieved(song: SongRecord, document: str, score: float) -> RetrievedSong:
    return RetrievedSong(
        song_id=song.song_id,
        title=song.title,
        artist=song.artist,
        album=song.album,
        release_year=song.release_year,
        genres=list(song.genres),
        tags=list(song.tags),
        score=score,
        document_text=document,
    )


def retrieve_bm25(
    query: str,
    top_k: int,
    *,
    songs: list[SongRecord] | None = None,
) -> list[RetrievedSong]:
    """Rank the unchanged song documents with BM25Okapi."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    catalog = songs if songs is not None else load_catalog(DEFAULT_CATALOG_PATH)
    documents = [song_to_document(song) for song in catalog]
    bm25 = BM25Okapi([tokenize(document) for document in documents])
    scores = bm25.get_scores(tokenize(query))
    ranked_indices = sorted(
        range(len(catalog)),
        key=lambda index: (-float(scores[index]), catalog[index].song_id),
    )[:top_k]
    results = []
    for index in ranked_indices:
        score = float(scores[index])
        if not math.isfinite(score):
            raise ValueError(f"BM25 returned a non-finite score: {score}")
        results.append(_song_to_retrieved(catalog[index], documents[index], score))
    return results


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedSong]],
    *,
    rrf_constant: int = HYBRID_RRF_CONSTANT,
) -> dict[str, float]:
    """Fuse rankings by position; raw cosine and BM25 scales never mix."""

    if rrf_constant <= 0:
        raise ValueError("rrf_constant must be positive.")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, song in enumerate(ranking, start=1):
            if song.song_id in seen:
                continue
            seen.add(song.song_id)
            scores[song.song_id] = scores.get(song.song_id, 0.0) + 1.0 / (
                rrf_constant + rank
            )
    return scores


def matches_hard_constraints(
    song: RetrievedSong,
    constraints: QueryConstraints,
) -> bool:
    """Apply hard year/artist filters; unknown years fail an explicit year filter."""

    if constraints.year_min is not None:
        if song.release_year is None or song.release_year < constraints.year_min:
            return False
    if constraints.year_max is not None:
        if song.release_year is None or song.release_year > constraints.year_max:
            return False
    if constraints.artist is not None:
        if song.artist.casefold() != constraints.artist.casefold():
            return False
    return True


def retrieve_hybrid(
    query: str,
    top_k: int = 10,
) -> list[HybridRetrievedSong]:
    """Return RRF-fused dense/BM25 results after deterministic hard filtering."""

    return retrieve_hybrid_with_details(query, top_k=top_k).hybrid


def retrieve_hybrid_with_details(
    query: str,
    top_k: int = 10,
) -> HybridRetrievalResult:
    """Expose all Phase 5 rankings, filtering decisions, and simple timings."""

    if not query.strip():
        raise ValueError("query must be a non-empty string.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    songs = load_catalog(DEFAULT_CATALOG_PATH)
    candidate_k = len(songs)
    constraints = parse_query_constraints(
        query, known_artists=sorted({song.artist for song in songs})
    )

    dense, dense_timing = retrieve_with_timing(query, top_k=candidate_k)
    bm25_started = perf_counter()
    bm25 = retrieve_bm25(query, candidate_k, songs=songs)
    bm25_seconds = perf_counter() - bm25_started

    fusion_started = perf_counter()
    fusion_scores = reciprocal_rank_fusion([dense, bm25])
    dense_by_id = {song.song_id: song for song in dense}
    bm25_by_id = {song.song_id: song for song in bm25}
    dense_ranks = {song.song_id: rank for rank, song in enumerate(dense, 1)}
    bm25_ranks = {song.song_id: rank for rank, song in enumerate(bm25, 1)}

    fused_ids = sorted(
        fusion_scores,
        key=lambda song_id: (
            -fusion_scores[song_id],
            dense_ranks.get(song_id, candidate_k + 1),
            bm25_ranks.get(song_id, candidate_k + 1),
            song_id,
        ),
    )
    hybrid: list[HybridRetrievedSong] = []
    filtered_song_ids: list[str] = []
    for song_id in fused_ids:
        source = dense_by_id.get(song_id) or bm25_by_id[song_id]
        if not matches_hard_constraints(source, constraints):
            filtered_song_ids.append(song_id)
            continue
        dense_song = dense_by_id.get(song_id)
        bm25_song = bm25_by_id.get(song_id)
        fusion_score = fusion_scores[song_id]
        hybrid.append(
            HybridRetrievedSong(
                song_id=source.song_id,
                title=source.title,
                artist=source.artist,
                album=source.album,
                release_year=source.release_year,
                genres=list(source.genres),
                tags=list(source.tags),
                score=fusion_score,
                document_text=source.document_text,
                dense_rank=dense_ranks.get(song_id),
                dense_score=dense_song.score if dense_song else None,
                bm25_rank=bm25_ranks.get(song_id),
                bm25_score=bm25_song.score if bm25_song else None,
                fusion_score=fusion_score,
            )
        )
        if len(hybrid) == top_k:
            break
    fusion_seconds = perf_counter() - fusion_started

    return HybridRetrievalResult(
        constraints=constraints,
        dense=dense[:top_k],
        bm25=bm25[:top_k],
        hybrid=hybrid,
        filtered_song_ids=filtered_song_ids,
        timing=HybridTiming(
            query_embedding_seconds=dense_timing.query_embedding_seconds,
            milvus_search_seconds=dense_timing.milvus_search_seconds,
            bm25_seconds=bm25_seconds,
            fusion_and_filter_seconds=fusion_seconds,
        ),
    )
