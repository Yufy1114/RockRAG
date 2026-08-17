"""Inspect BGE vectors and run direct dot-product retrieval sanity checks."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rockrag import embed_documents, embed_query, load_catalog, song_to_document  # noqa: E402
from rockrag.embeddings import MODEL_NAME, load_model  # noqa: E402


QUERIES = [
    "melodic 80s hard rock power ballad",
    "classic traditional heavy metal from the 1980s",
    "fast aggressive thrash metal",
    "progressive melodic metal from the 1990s",
]


def main() -> None:
    load_started = perf_counter()
    model = load_model()
    load_seconds = perf_counter() - load_started

    songs = load_catalog()
    documents = [song_to_document(song) for song in songs]

    batch_started = perf_counter()
    song_embeddings = embed_documents(documents)
    batch_seconds = perf_counter() - batch_started

    example_index = next(
        index
        for index, song in enumerate(songs)
        if song.song_id == "scorpions-still-loving-you"
    )
    example = songs[example_index]
    example_embedding = song_embeddings[example_index]

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {model.device}")
    print(f"Model download/load time: {load_seconds:.6f} seconds")
    print(f"38-song batch embedding time: {batch_seconds:.6f} seconds")
    print("\nSong:")
    print(f"{example.title} — {example.artist}")
    print("\nDocument text:")
    print(documents[example_index])
    print(f"\nEmbedding shape: {example_embedding.shape}")
    print("First 10 embedding values:")
    print(np.array2string(example_embedding[:10], precision=8, separator=", "))
    print(f"L2 norm: {np.linalg.norm(example_embedding):.9f}")

    print("\nFull catalog:")
    print(f"Number of songs: {len(songs)}")
    print(f"Embedding matrix shape: {song_embeddings.shape}")
    print(f"dtype: {song_embeddings.dtype}")
    print(f"NaN count: {np.isnan(song_embeddings).sum()}")
    print(f"Inf count: {np.isinf(song_embeddings).sum()}")

    query_times: list[float] = []
    for query in QUERIES:
        query_started = perf_counter()
        query_embedding = embed_query(query)
        query_times.append(perf_counter() - query_started)

        # Both sides are normalized, so this dot product is cosine similarity.
        similarities = song_embeddings @ query_embedding
        top_indices = np.argsort(similarities)[::-1][:5]

        print(f"\nQuery: {query}")
        print(f"Query embedding time: {query_times[-1]:.6f} seconds")
        for rank, index in enumerate(top_indices, start=1):
            song = songs[int(index)]
            print(
                f"{rank}. {song.title} — {song.artist} | "
                f"{float(similarities[index]):.6f}"
            )

    print(f"\nMean single-query embedding time: {np.mean(query_times):.6f} seconds")


if __name__ == "__main__":
    main()
