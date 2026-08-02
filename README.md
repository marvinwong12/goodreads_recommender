# Sci-Fi/Fantasy Book Recommender

A two-stage book recommender trained on Goodreads interaction data: a
multi-channel retrieval stage narrows the full catalog down to a candidate
pool, and a learned XGBoost reranker sorts that pool into a final top-10.

**Live demo**: https://scifi-fantasy-recommender-1093855474171.us-central1.run.app

Search for a few sci-fi/fantasy books you've enjoyed and get live
recommendations.

## How it works

**Stage 1 — retrieval.** Three independent channels each propose candidates,
unioned into one pool:

- **LightGCN** — a graph neural collaborative-filtering signal, trained on a
  k-core user-book interaction graph via BPR loss with popularity-weighted
  negative sampling.
- **Semantic similarity** — book description embeddings (sentence
  transformers + FAISS), compared against a recency-weighted profile of a
  user's own reading history.
- **Item-item co-occurrence** — "readers of your history also read...",
  a sparse book-book similarity matrix built from interaction co-occurrence.

The LightGCN and semantic signals are combined via a dynamic fusion weight
that shifts toward content-based similarity for newer users and toward
collaborative filtering as a user's read history grows.

**Stage 2 — reranking.** An XGBoost learning-to-rank model (`rank:ndcg`,
monotonic constraints on the retrieval scores) reranks the unioned candidate
pool using the three retrieval scores plus author-affinity and book-popularity
features.

**Cold start.** A user who isn't in the training graph (e.g. someone typing
books into the live demo) still gets the full pipeline: their LightGCN vector
is approximated by folding in the mean embedding of the books they entered,
while the semantic and co-occurrence channels need no approximation at all.

## Results

Evaluated on a time-based held-out test split (not random leave-one-out):

| | HR@5 | HR@10 | NDCG@10 | MRR@10 |
|---|---|---|---|---|
| Stage 1 only (fused retrieval, no reranking) | 0.1152 | 0.1625 | 0.0667 | 0.0778 |
| Full two-stage pipeline | **0.2167** | **0.2910** | **0.1244** | **0.1485** |

See [POSTMORTEM.md](POSTMORTEM.md) for
the debugging journey, including a stretch where it was making
recommendations *worse* than not reranking at all.

## Tech stack

PyTorch (LightGCN) · sentence-transformers + FAISS (semantic embeddings) ·
XGBoost + Optuna (reranker) · scipy sparse (item-item co-occurrence) ·
PySpark (graph/data prep) · FastAPI + vanilla JS (serving) · Docker + Google
Cloud Run (deployment)

## Repo layout

```
src/
  data/         Spark jobs: cleaning, k-core graph construction, ranker dataset build
  models/       LightGCN training, semantic embedding, item-cf, XGBoost ranker, dynamic fusion
  evaluation/   Offline test-set evaluation (fused baseline vs. full pipeline)
  inference/    Live recommendation path for new/cold-start users
app/            FastAPI app + static frontend for the live demo
deploy/         Docker build context and staging script for deployment
```

## Running locally

```bash
pip install -r requirements.txt

# Rebuild the data/model artifacts, in order:
python src/data/01_etl.py
python src/data/02_clean_data.py
python src/data/03_make_graph_data.py
python src/models/train_lightgcn.py
python src/models/embed_books.py
python src/models/item_cf.py
python src/data/04_build_ranker_dataset.py
python src/models/train_xgboost.py

# Evaluate:
python src/evaluation/eval_test_set.py

# Run the demo app:
uvicorn app.main:app --reload
```

To build and run the containerized version:

```bash
./deploy/prepare_deploy.sh
cd deploy && docker build -t goodreads-recommender . && docker run -p 8000:8080 goodreads-recommender
```
