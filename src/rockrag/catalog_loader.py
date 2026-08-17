"""Load and validate the normalized RockRAG song catalog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_CATALOG_PATH
from .models import SongRecord


class CatalogValidationError(ValueError):
    """Raised when catalog data is malformed or internally inconsistent."""


def _require_non_empty_string(value: Any, field_name: str, location: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CatalogValidationError(
            f"{location}: '{field_name}' must be a non-empty string."
        )


def _validate_optional_string(value: Any, field_name: str, location: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise CatalogValidationError(
            f"{location}: '{field_name}' must be null or a non-empty string."
        )


def _validate_string_list(value: Any, field_name: str, location: str) -> None:
    if not isinstance(value, list):
        raise CatalogValidationError(f"{location}: '{field_name}' must be a list.")
    for item_index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise CatalogValidationError(
                f"{location}: '{field_name}[{item_index}]' must be a "
                "non-empty string."
            )


def _validate_record(data: Any, index: int) -> SongRecord:
    location = f"songs[{index}]"
    if not isinstance(data, dict):
        raise CatalogValidationError(f"{location} must be a JSON object.")

    for field_name in ("song_id", "title", "artist"):
        _require_non_empty_string(data.get(field_name), field_name, location)

    release_year = data.get("release_year")
    if release_year is not None and (
        isinstance(release_year, bool) or not isinstance(release_year, int)
    ):
        raise CatalogValidationError(
            f"{location}: 'release_year' must be an integer or null, "
            f"got {type(release_year).__name__}."
        )

    for field_name in ("genres", "tags"):
        _validate_string_list(data.get(field_name), field_name, location)

    optional_strings = (
        "album",
        "musicbrainz_recording_id",
        "musicbrainz_release_group_id",
        "isrc",
        "metadata_source",
        "tags_source",
        "verified_at",
    )
    for field_name in optional_strings:
        _validate_optional_string(data.get(field_name), field_name, location)

    try:
        return SongRecord.from_dict(data)
    except TypeError as exc:
        raise CatalogValidationError(f"{location}: invalid fields: {exc}") from exc


def load_catalog(path: str | Path = DEFAULT_CATALOG_PATH) -> list[SongRecord]:
    """Read a JSON catalog, validate it, and return SongRecord objects."""

    catalog_path = Path(path)
    try:
        raw_data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogValidationError(
            f"Catalog file does not exist: {catalog_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CatalogValidationError(
            f"Catalog is not valid JSON: {catalog_path} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc

    if not isinstance(raw_data, list):
        raise CatalogValidationError("Catalog root must be a JSON list of songs.")

    songs: list[SongRecord] = []
    seen_ids: dict[str, int] = {}
    for index, item in enumerate(raw_data):
        song = _validate_record(item, index)
        if song.song_id in seen_ids:
            first_index = seen_ids[song.song_id]
            raise CatalogValidationError(
                f"songs[{index}]: duplicate song_id '{song.song_id}' "
                f"(first used at songs[{first_index}])."
            )
        seen_ids[song.song_id] = index
        songs.append(song)

    return songs
