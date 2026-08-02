# Sci-Fi/Fantasy Book Recommender

A two-stage book recommender trained on Goodreads interaction data: a
multi-channel retrieval stage narrows the full catalog down to a candidate
pool, and a learned XGBoost reranker sorts that pool into a final top-10.

**Live demo**: https://scifi-fantasy-recommender-1093855474171.us-central1.run.app

Search for a few sci-fi/fantasy books you've enjoyed and get live
recommendations.

## Data

Trained on the UCSD Goodreads interaction and review datasets:

> Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior
> Chains", in RecSys'18.
>
> Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained
> Spoiler Detection from Large-Scale Review Corpora", in ACL'19.

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

## Next steps / open questions

This is a first working pipeline, not a finished one. Preliminary directions
worth pursuing next:

**Data**
- Expand dataset to include whole catalog of genres, see if recommendation
pipeline still holds up.

**Modeling**
- Replace the recency-*weighted* semantic profile (an exponential-decay
  approximation of "recent taste") with a real sequence model (SASRec /
  GRU4Rec / BERT4Rec) that learns order-dependent patterns directly, e.g.
  genre binges or working through a series in order.
- Add a fourth retrieval channel (ie. popularity/trending) and check whether
  it recalls a meaningfully different slice of relevant candidates than the
  three existing channels, the way co-occurrence did.

**Features**
- Investigate whether `book_average_rating` / `book_log_ratings_count`
  should be interacted with user tenure (a new user may weight raw
  popularity differently than an established one).

**Evaluation**
- All results are evaluated at this dataset's catalog scale (~110K books);
  worth checking how recall/ranking quality degrades (or doesn't) as
  `STAGE_1_TOP_K` and catalog size scale further.

**Productionization**
- Real users' interactions currently never update embeddings live. A
  scheduled retrain is the realistic path, but it's worth defining how
  often, and what a staleness/monitoring signal for "the graph needs
  retraining" would look like.
- No exposure/position-bias correction or diversity/business-rules layer
