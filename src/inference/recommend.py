# src/inference/recommend.py
"""
Live recommendation path for a "new" user who isn't in the training graph:
they type in books they've enjoyed, and get real-time recommendations
through the same two-stage pipeline (fused + co-occurrence retrieval,
XGBoost reranking) used for offline evaluation.

The one approximation versus an existing user: LightGCN has no learned
embedding for someone outside the graph. We fold them in by averaging the
LightGCN *book* embeddings of what they entered (LightGCN's own propagation
is neighbor-averaging, so this is a reasonable zero-layer approximation
without retraining the graph). The semantic and co-occurrence channels need
no approximation -- both are computable directly from the entered books.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.dynamic_fusion import fuse_scores, recency_weighted_profile
from src.models.item_cf import load_item_similarity, score_candidates as score_cooccurrence_candidates

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
LIGHTGCN_DIR = PROJECT_ROOT / "models" / "lightgcn"
VECTOR_DB_DIR = PROJECT_ROOT / "models" / "vector_index"
XGBOOST_DIR = PROJECT_ROOT / "models" / "xgboost"

STAGE_1_TOP_K = 200
DEFAULT_USER_AVG_RATING = 3.8  # used when the user doesn't supply ratings for their input books

FEATURE_NAMES = [
    'lightgcn_score', 'semantic_score', 'fused_score',
    'user_avg_rating', 'user_explicit_rating_count',
    'author_read_count', 'book_average_rating', 'book_log_ratings_count',
    'cooccurrence_score',
]


class RecommenderService:
    """Loads every artifact once; call .recommend(...) per request."""

    def __init__(self):
        print("Loading catalog and mappings...")
        self.books = pd.read_parquet(PROCESSED_DIR / "books_clean.parquet")
        self.book_map = pd.read_parquet(GRAPH_DIR / "book_mapping.parquet")

        self.num_books = len(self.book_map)
        self.book_id_to_idx = dict(zip(self.book_map['book_id'], self.book_map['book_idx']))
        self.book_idx_to_id = dict(zip(self.book_map['book_idx'], self.book_map['book_id']))

        print("Loading LightGCN book embeddings...")
        self.book_embs_gcn = np.load(LIGHTGCN_DIR / "book_embeddings.npy")

        print("Loading and aligning semantic embeddings...")
        sem_embs_raw = np.load(VECTOR_DB_DIR / "book_embeddings.npy")
        sem_ids_raw = np.load(VECTOR_DB_DIR / "book_ids.npy", allow_pickle=True)
        aligned = np.zeros((self.num_books, sem_embs_raw.shape[1]))
        for i, b_id in enumerate(sem_ids_raw):
            b_id_str = str(b_id)
            if b_id_str in self.book_id_to_idx:
                aligned[self.book_id_to_idx[b_id_str]] = sem_embs_raw[i]
        norms = np.linalg.norm(aligned, axis=1, keepdims=True)
        self.aligned_sem_embs = np.divide(aligned, norms, out=np.zeros_like(aligned), where=norms != 0)

        print("Building static per-book feature arrays...")
        book_meta = self.book_map.merge(
            self.books[['book_id', 'title', 'primary_author_id', 'average_rating', 'ratings_count']],
            on='book_id', how='left'
        )
        self.author_id_arr = np.full(self.num_books, None, dtype=object)
        self.title_arr = np.full(self.num_books, None, dtype=object)
        self.avg_rating_arr = np.full(self.num_books, self.books['average_rating'].mean(), dtype=np.float64)
        self.log_ratings_count_arr = np.zeros(self.num_books, dtype=np.float64)
        for row in book_meta.itertuples(index=False):
            b_idx = int(row.book_idx)
            self.author_id_arr[b_idx] = row.primary_author_id
            self.title_arr[b_idx] = row.title
            if pd.notnull(row.average_rating):
                self.avg_rating_arr[b_idx] = row.average_rating
            if pd.notnull(row.ratings_count):
                self.log_ratings_count_arr[b_idx] = np.log1p(row.ratings_count)

        print("Loading item-item co-occurrence matrix...")
        self.item_sim = load_item_similarity()

        print("Loading trained XGBoost ranker...")
        self.ranker = xgb.XGBRanker()
        self.ranker.load_model(XGBOOST_DIR / "ranker_model.json")

        # Search index for the title-autocomplete endpoint.
        self._search_df = self.books[['book_id', 'title', 'ratings_count']].copy()
        self._search_df['title_lower'] = self._search_df['title'].str.lower()

        print("✓ RecommenderService ready.")

    def search_books(self, query: str, limit: int = 10):
        """Simple substring search over titles, most-rated first -- good
        enough for an autocomplete box without standing up real text search."""
        q = query.lower().strip()
        if not q:
            return []
        matches = self._search_df[self._search_df['title_lower'].str.contains(q, na=False, regex=False)]
        matches = matches.sort_values('ratings_count', ascending=False).head(limit)
        return matches[['book_id', 'title']].to_dict('records')

    def recommend(self, book_ids: list[str], user_avg_rating: float = None, top_k: int = 10):
        """
        book_ids: the user's enjoyed books, oldest -> most recent (order lets
                  the semantic profile weight recent taste more heavily; if
                  you don't have ordering, passing them in any order still
                  works, just without the recency benefit).
        """
        book_idx_list = [self.book_id_to_idx[b] for b in book_ids if b in self.book_id_to_idx]
        if not book_idx_list:
            raise ValueError("None of the provided book_ids were found in the catalog.")

        u_count = len(book_idx_list)
        u_avg_rating = user_avg_rating if user_avg_rating is not None else DEFAULT_USER_AVG_RATING

        # Books already read, keyed by (title, author) rather than book_id --
        # the catalog has multiple editions of the same book as separate
        # entries, and a different edition of something the user just told us
        # they read is not a recommendation.
        read_keys = {
            ((self.title_arr[b_idx] or "").strip().lower(), self.author_id_arr[b_idx])
            for b_idx in book_idx_list
        }

        author_read_counts = Counter(
            a for a in (self.author_id_arr[i] for i in book_idx_list) if a is not None
        )

        # --- Fold-in LightGCN vector: mean of the entered books' embeddings ---
        user_vec_gcn = self.book_embs_gcn[book_idx_list].mean(axis=0)
        lightgcn_scores = np.dot(self.book_embs_gcn, user_vec_gcn)

        # --- Semantic profile, recency-weighted over entered books ---
        user_sem_profile = recency_weighted_profile(self.aligned_sem_embs, book_idx_list)
        semantic_scores = np.dot(self.aligned_sem_embs, user_sem_profile)

        fused_scores, _ = fuse_scores(lightgcn_scores, semantic_scores, u_count)
        cooccurrence_scores = score_cooccurrence_candidates(self.item_sim, book_idx_list)

        # Never recommend back one of the input books.
        lightgcn_scores[book_idx_list] = -np.inf
        semantic_scores[book_idx_list] = -np.inf
        fused_scores[book_idx_list] = -np.inf
        cooccurrence_scores[book_idx_list] = -np.inf

        fused_top_k = np.argpartition(fused_scores, -STAGE_1_TOP_K)[-STAGE_1_TOP_K:]
        cooc_top_k = np.argpartition(cooccurrence_scores, -STAGE_1_TOP_K)[-STAGE_1_TOP_K:]
        candidate_indices = np.array(list(set(fused_top_k) | set(cooc_top_k)))
        n_candidates = len(candidate_indices)

        c_author_read = np.array([
            author_read_counts.get(self.author_id_arr[b_idx], 0) for b_idx in candidate_indices
        ])
        X = np.column_stack((
            lightgcn_scores[candidate_indices],
            semantic_scores[candidate_indices],
            fused_scores[candidate_indices],
            np.full(n_candidates, u_avg_rating),
            np.full(n_candidates, u_count),
            c_author_read,
            self.avg_rating_arr[candidate_indices],
            self.log_ratings_count_arr[candidate_indices],
            cooccurrence_scores[candidate_indices],
        ))
        X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
        ranker_scores = self.ranker.predict(X_df)

        # The catalog has multiple editions of the same book as separate
        # entries (same title+author, different book_id) -- dedupe so they
        # don't compete for candidate slots that should only be filled once.
        # Pull a wider slice than top_k since some will collapse together.
        order = np.argsort(ranker_scores)[::-1][:top_k * 3]

        seen = set()
        results = []
        for i in order:
            b_idx = candidate_indices[i]
            key = (
                (self.title_arr[b_idx] or "").strip().lower(),
                self.author_id_arr[b_idx],
            )
            if key in seen or key in read_keys:
                continue
            seen.add(key)
            results.append({
                "book_id": self.book_idx_to_id[b_idx],
                "title": self.title_arr[b_idx],
                "score": float(ranker_scores[i]),
            })
            if len(results) == top_k:
                break

        return results


if __name__ == "__main__":
    service = RecommenderService()

    print("\nSearching for 'hobbit'...")
    for r in service.search_books("hobbit", limit=5):
        print(" ", r)

    hits = service.search_books("hobbit", limit=1)
    if hits:
        print(f"\nRecommendations based on '{hits[0]['title']}':")
        for rec in service.recommend([hits[0]['book_id']], top_k=10):
            print(f"  {rec['score']:.4f}  {rec['title']}")
