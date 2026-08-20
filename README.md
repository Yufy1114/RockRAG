# RockRAG

LLM-powered Hard Rock and Heavy Metal music discovery using retrieval-first recommendations.

## Current status

The current retrieval-first pipeline is:

```text
Song Catalog
    ↓
Song Documents
    ↓
BGE Embeddings
    ↓
Milvus Lite
    + BM25 over the same song documents
    ↓
Reciprocal Rank Fusion
    ↓
Hard year/artist metadata filtering
    ↓
cross-encoder/ms-marco-MiniLM-L6-v2 reranking
    ↓
Configurable LLM: local Ollama/Qwen or Google Gemini
    ↓
Grounded Recommendations
```

## Agent layer

Phase 7 adds a single Ollama/Qwen agent using native tool calling. Tool
selection comes from `response.message.tool_calls`; the application only
validates and dispatches calls from an explicit registry.

```text
User
  → Qwen Agent
  → RockRAG Tools
      ├── search_songs
      ├── get_song
      ├── compare_songs
      └── build_playlist
  → Hybrid Retrieval / CrossEncoder when requested by a tool
  → auditable tool trace
  → final response
```

The agent is bounded by `MAX_AGENT_STEPS=5`. It has no shell, file, web,
Spotify, or external music API tools and is not a multi-agent system.

The retrieval stages remain explicit:

```text
Query constraints (hard: year/artist; soft: genre words)
    + Dense BGE/Milvus ranking
    + BM25 ranking over the unchanged song documents
    → Reciprocal Rank Fusion
    → hard metadata filtering
    → CrossEncoder reranking of Hybrid candidates only
```

The generation provider is selected with `LLM_PROVIDER=ollama` or `gemini`.
Generated song IDs are validated against the final retrieved candidates, and
all factual display fields are backfilled from `RetrievedSong`.

## Retrieval evaluation

The curated evaluation set contains 15 Hard Rock/Metal queries with manually
graded qrels (`2=highly relevant`, `1=relevant`). The labels use catalog
metadata and curator taxonomy, not model-generated judgments. Metrics have
different purposes: Recall@K checks whether relevant songs were retrieved,
MRR rewards an early first relevant result, and nDCG rewards highly relevant
songs appearing near the top.

Actual warmed-up results:

| Retrieval system | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | P@5 | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0.6179 | 0.7701 | 0.8667 | 0.6872 | 0.7246 | 0.4400 | 0.013496 s |
| Hybrid | 0.7318 | 0.9373 | 0.9667 | 0.8729 | 0.9096 | 0.5867 | 0.016298 s |
| Hybrid + CrossEncoder | 0.8051 | 0.9373 | 1.0000 | 0.9756 | 0.9698 | 0.6667 | 0.029854 s |

The reranker cold load measured 11.027410 seconds and is excluded from the
warmed-up latency table. Full per-query rankings and failure analysis are in
`evaluation/results/latest.json`.

Future work includes larger-catalog evaluation, reason-grounding checks, and
application APIs. UI/API and multi-agent layers are not implemented.

## Structure

- `data/raw`: auditable source snapshots.
- `data/processed`: normalized song catalog.
- `src/rockrag`: catalog models and transformation logic.
- `scripts`: small executable inspection tools.
- `evaluation`: curated qrels, transparent metrics, runner, and saved results.
- `tests`: unit and opt-in integration tests across Phases 1–7.
- `storage`: local Milvus Lite database and ignored model cache.

## Catalog source

Factual recording metadata is selected from a small, rate-limited MusicBrainz API snapshot. Genres and tags retain explicit provenance and missing values are not inferred.
