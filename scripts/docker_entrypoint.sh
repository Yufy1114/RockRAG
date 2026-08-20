#!/bin/sh
set -eu

# Milvus Lite is a local file store. Build it from the immutable catalog when
# the mounted storage volume is empty; repeated starts reuse the existing DB.
if [ ! -e /app/storage/rockrag.db ]; then
  echo "Milvus Lite index not found; building it from data/processed/songs.json..."
  python scripts/build_index.py
fi

exec uvicorn rockrag.api:app \
  --host "${WEB_HOST:-0.0.0.0}" \
  --port "${WEB_PORT:-8000}"
