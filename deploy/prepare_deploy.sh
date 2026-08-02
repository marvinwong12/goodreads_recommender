#!/usr/bin/env bash
# Stages exactly the files the serving app needs into deploy/, so the
# Docker build context is small and doesn't drag in the ~16GB of training
# data/checkpoints that live alongside them in the main repo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY="$ROOT/deploy"

mkdir -p \
  "$DEPLOY/src/models" \
  "$DEPLOY/src/inference" \
  "$DEPLOY/app/static" \
  "$DEPLOY/data/processed/graph" \
  "$DEPLOY/models/lightgcn" \
  "$DEPLOY/models/vector_index" \
  "$DEPLOY/models/item_cf" \
  "$DEPLOY/models/xgboost"

# Code (only what src/inference/recommend.py actually imports)
cp "$ROOT/src/models/dynamic_fusion.py" "$DEPLOY/src/models/"
cp "$ROOT/src/models/item_cf.py" "$DEPLOY/src/models/"
cp "$ROOT/src/inference/recommend.py" "$DEPLOY/src/inference/"
cp "$ROOT/app/main.py" "$DEPLOY/app/"
cp "$ROOT/app/static/index.html" "$DEPLOY/app/static/"

# Model/data artifacts needed at inference time
# (book_mapping.parquet is Spark output -- a directory of part files, not a
# single file -- pandas.read_parquet reads it transparently)
rm -rf "$DEPLOY/data/processed/books_clean.parquet" "$DEPLOY/data/processed/graph/book_mapping.parquet"
cp -r "$ROOT/data/processed/graph/book_mapping.parquet" "$DEPLOY/data/processed/graph/"

# books_clean.parquet is trimmed to only the columns RecommenderService
# actually reads (description/text_for_embedding/image_url/author_ids are the
# bulk of its 287MB and are never touched at serving time) -- this is the
# single biggest lever on the container's memory footprint.
python3 - "$ROOT/data/processed/books_clean.parquet" "$DEPLOY/data/processed/books_clean.parquet" <<'EOF'
import sys
import pandas as pd
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_parquet(src, columns=['book_id', 'title', 'primary_author_id', 'average_rating', 'ratings_count'])
df.to_parquet(dst, index=False)
print(f"  books_clean.parquet trimmed: {len(df):,} rows, columns={list(df.columns)}")
EOF
cp "$ROOT/models/lightgcn/book_embeddings.npy" "$DEPLOY/models/lightgcn/"
cp "$ROOT/models/vector_index/book_embeddings.npy" "$DEPLOY/models/vector_index/"
cp "$ROOT/models/vector_index/book_ids.npy" "$DEPLOY/models/vector_index/"
cp "$ROOT/models/item_cf/item_similarity.npz" "$DEPLOY/models/item_cf/"
cp "$ROOT/models/xgboost/ranker_model.json" "$DEPLOY/models/xgboost/"

echo "✓ Staged $(du -sh "$DEPLOY" | cut -f1) into $DEPLOY"
