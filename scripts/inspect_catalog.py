"""Print Phase 1 catalog quality statistics and document examples."""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag import load_catalog, song_to_document  # noqa: E402


def main() -> None:
    songs = load_catalog()
    duplicate_count = len(songs) - len({song.song_id for song in songs})
    years = [song.release_year for song in songs if song.release_year is not None]
    genre_counts = Counter(genre for song in songs for genre in song.genres)

    print(f"Loaded {len(songs)} songs")
    print(f"Duplicate song IDs: {duplicate_count}")
    print(f"Total songs: {len(songs)}")
    print(f"Unique artists: {len({song.artist for song in songs})}")
    print(f"Release year range: {min(years)}-{max(years)}" if years else "Release year range: missing")
    print("Genre distribution:")
    for genre, count in sorted(genre_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {genre}: {count}")
    print(f"Missing album count: {sum(song.album is None for song in songs)}")
    print(f"Missing year count: {sum(song.release_year is None for song in songs)}")
    print(
        "Missing MusicBrainz ID count: "
        f"{sum(song.musicbrainz_recording_id is None for song in songs)}"
    )

    example = random.Random(42).choice(songs)
    print("\nExample SongRecord:")
    print(example)
    print("\nEmbedding document text:")
    print(song_to_document(example))


if __name__ == "__main__":
    main()
