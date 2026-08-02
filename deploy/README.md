---
title: Goodreads Recommender
emoji: 📚
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Goodreads Recommender

Search for a few sci-fi/fantasy books you've enjoyed and get live
recommendations from a two-stage retrieval + reranking pipeline:

- **Retrieval**: a LightGCN collaborative-filtering signal, a semantic
  (content embedding) signal, and an item-item co-occurrence channel, each
  contributing independent candidates.
- **Reranking**: an XGBoost learning-to-rank model combines all channels'
  scores plus author-affinity and book-popularity features, under monotonic
  constraints on the retrieval scores.

New users (not in the training graph) are handled via embedding fold-in: the
LightGCN vector is approximated as the mean of the entered books' embeddings,
while the semantic and co-occurrence channels need no approximation at all.
