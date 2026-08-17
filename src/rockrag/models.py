"""Data models for factual song catalog records."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SongRecord:
    """One recording used as the retrieval unit in RockRAG.

    Only ``title`` and ``artist`` are universally required facts. The loader
    performs the stricter catalog-level checks, including unique ``song_id``.
    Missing facts stay as ``None`` or an empty list; they are never inferred.
    """

    song_id: str
    title: str
    artist: str
    album: str | None = None
    release_year: int | None = None
    genres: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    musicbrainz_recording_id: str | None = None
    musicbrainz_release_group_id: str | None = None
    isrc: str | None = None
    metadata_source: str | None = None
    tags_source: str | None = None
    verified_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SongRecord":
        """Construct a record without silently accepting unknown fields."""

        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)
