"""Convert a SongRecord into the exact text embedded by later phases."""

from .models import SongRecord


def song_to_document(song: SongRecord) -> str:
    """Build deterministic factual text, omitting every missing field."""

    lines = [
        f"Title: {song.title}",
        f"Artist: {song.artist}",
    ]

    if song.album:
        lines.append(f"Album: {song.album}")
    if song.release_year is not None:
        lines.append(f"Release year: {song.release_year}")
    if song.genres:
        lines.append(f"Genres: {', '.join(song.genres)}")
    if song.tags:
        lines.append(f"Tags: {', '.join(song.tags)}")

    return "\n".join(lines)
