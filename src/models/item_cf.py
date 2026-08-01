# src/models/item_cf.py
"""
Item-item co-occurrence retrieval channel ("users who read A also read B"),
built as a second, independent candidate-generation channel alongside the
LightGCN + semantic fused score. Production retrieval stacks (Amazon's
original item-to-item CF, most multi-channel recall systems) union several
cheap, structurally-different retrieval channels rather than leaning on one
blended score, since each channel recalls a different slice of relevant
items.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
ITEM_CF_DIR = PROJECT_ROOT / "models" / "item_cf"

TOP_N_NEIGHBORS = 50

# Caps the per-user history used to build the book-book co-occurrence matrix.
# Matmul cost scales with sum(user_degree^2), and a handful of power users
# with thousands of reads would otherwise dominate that cost; using each
# user's most recent MAX_HISTORY_PER_USER reads keeps the matrix cheap to
# build while still capturing current taste.
MAX_HISTORY_PER_USER = 100


def build_item_similarity():
    ITEM_CF_DIR.mkdir(parents=True, exist_ok=True)

    with open(GRAPH_DIR / "graph_meta.json") as f:
        meta = json.load(f)
    num_users = meta["num_users"]
    num_books = meta["num_books"]

    print("Loading leakage-free history edges for item-item co-occurrence...")
    interactions = pd.read_parquet(PROCESSED_DIR / "interactions_clean.parquet")
    user_map = pd.read_parquet(GRAPH_DIR / "user_mapping.parquet")
    book_map = pd.read_parquet(GRAPH_DIR / "book_mapping.parquet")

    # Same leakage rule as the LightGCN graph (03_make_graph_data.py): only
    # is_gcn_fit history, never the held-out ranker-label window.
    history = interactions[
        (interactions["is_gcn_fit"] == True) & (interactions["is_engagement"] == 1)
    ].merge(user_map, on="user_id").merge(book_map, on="book_id")

    history = history.sort_values(["user_idx", "interaction_date"])
    history = history.groupby("user_idx").tail(MAX_HISTORY_PER_USER)

    rows = history["user_idx"].to_numpy()
    cols = history["book_idx"].to_numpy()
    data = np.ones(len(history), dtype=np.float32)
    R = sp.csr_matrix((data, (rows, cols)), shape=(num_users, num_books))

    print(f"Computing book-book co-occurrence over {len(history):,} (capped) edges...")
    C = (R.T @ R).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()

    print("Normalizing to cosine-like item similarity...")
    degree = np.asarray(R.sum(axis=0)).ravel()
    inv_sqrt_deg = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt_deg[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    D = sp.diags(inv_sqrt_deg)
    S = (D @ C @ D).tocsr()

    print(f"Pruning to top-{TOP_N_NEIGHBORS} neighbors per book...")
    S = _prune_top_n(S, TOP_N_NEIGHBORS)

    out_path = ITEM_CF_DIR / "item_similarity.npz"
    sp.save_npz(out_path, S)
    print(f"✓ Item-item similarity matrix saved to {out_path} (nnz={S.nnz:,})")


def _prune_top_n(matrix, n):
    matrix = matrix.tocsr()
    data, indices, indptr = [], [], [0]
    for row_idx in range(matrix.shape[0]):
        start, end = matrix.indptr[row_idx], matrix.indptr[row_idx + 1]
        row_data = matrix.data[start:end]
        row_indices = matrix.indices[start:end]
        if len(row_data) > n:
            keep = np.argpartition(row_data, -n)[-n:]
            row_data = row_data[keep]
            row_indices = row_indices[keep]
        data.append(row_data)
        indices.append(row_indices)
        indptr.append(indptr[-1] + len(row_data))
    return sp.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), indptr),
        shape=matrix.shape,
    )


_SIM_CACHE = None


def load_item_similarity():
    global _SIM_CACHE
    if _SIM_CACHE is None:
        _SIM_CACHE = sp.load_npz(ITEM_CF_DIR / "item_similarity.npz")
    return _SIM_CACHE


def score_candidates(sim_matrix, history_indices):
    """Sum of item-item similarity rows for a user's history books -> a
    dense co-occurrence relevance score over every book in the catalog."""
    num_books = sim_matrix.shape[0]
    if not history_indices:
        return np.zeros(num_books, dtype=np.float32)
    return np.asarray(sim_matrix[history_indices].sum(axis=0)).ravel()


if __name__ == "__main__":
    build_item_similarity()
