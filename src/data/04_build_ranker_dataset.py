import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.dynamic_fusion import fuse_scores

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
LIGHTGCN_DIR = PROJECT_ROOT / "models" / "lightgcn"
VECTOR_DB_DIR = PROJECT_ROOT / "models" / "vector_index"

STAGE_1_TOP_K = 100
MAX_DATASET_YEAR = 2017  # Anchors reference year to dataset max interaction year

def build_hard_negative_dataset():
    print("Loading datasets and mappings...")
    interactions = pd.read_parquet(PROCESSED_DIR / "interactions_clean.parquet")
    books = pd.read_parquet(PROCESSED_DIR / "books_clean.parquet")
    book_map = pd.read_parquet(GRAPH_DIR / "book_mapping.parquet")
    user_map = pd.read_parquet(GRAPH_DIR / "user_mapping.parquet")
    
    num_books = len(book_map)
    
    # 1. Load Embeddings
    print("Loading LightGCN Embeddings...")
    user_embs_gcn = np.load(LIGHTGCN_DIR / "user_embeddings.npy")
    book_embs_gcn = np.load(LIGHTGCN_DIR / "book_embeddings.npy")
    
    print("Loading Semantic Vector DB...")
    sem_embs_raw = np.load(VECTOR_DB_DIR / "book_embeddings.npy")
    sem_ids_raw = np.load(VECTOR_DB_DIR / "book_ids.npy", allow_pickle=True)
    
    aligned_sem_embs = np.zeros((num_books, sem_embs_raw.shape[1]))
    id_to_idx = {str(k): v for k, v in zip(book_map['book_id'], book_map['book_idx'])}
    
    for i, b_id in enumerate(sem_ids_raw):
        b_id_str = str(b_id)
        if b_id_str in id_to_idx:
            aligned_sem_embs[id_to_idx[b_id_str]] = sem_embs_raw[i]
            
    norms = np.linalg.norm(aligned_sem_embs, axis=1, keepdims=True)
    aligned_sem_embs = np.divide(aligned_sem_embs, norms, out=np.zeros_like(aligned_sem_embs), where=norms!=0)
    
    # 2. Prepare User Interaction Data
    #
    # --- LEAK FIX: disjoint history vs. label windows ---
    # `history_interactions` (is_gcn_fit) is the slice LightGCN and the semantic
    # user profile were built from. `label_interactions` is the later, disjoint
    # slice (is_train & ~is_gcn_fit) held back from stage 1 specifically so it
    # can be used as ranker labels. Because LightGCN/semantic scores were never
    # fit on label_interactions edges, lightgcn_score/semantic_score/fused_score
    # can no longer be a trivial readout of "was this edge trained on".
    train_interactions = interactions[interactions['is_train'] == True]
    history_interactions = train_interactions[
        (train_interactions['is_gcn_fit'] == True) & (train_interactions['is_engagement'] == 1)
    ].merge(user_map, on='user_id').merge(book_map, on='book_id')
    label_interactions = train_interactions[
        (train_interactions['is_gcn_fit'] == False) & (train_interactions['is_engagement'] == 1)
    ].merge(user_map, on='user_id').merge(book_map, on='book_id')

    user_metadata_dict = history_interactions[['user_idx', 'user_avg_rating', 'user_explicit_rating_count']].drop_duplicates().set_index('user_idx').to_dict('index')
    user_to_history = history_interactions.groupby('user_idx')['book_idx'].apply(list).to_dict()
    user_to_labels = label_interactions.groupby('user_idx')['book_idx'].apply(list).to_dict()

    # Reference year for book-age features: each user's most recent *history*
    # (pre-cutoff) interaction year, i.e. what would be knowable at candidate-
    # generation time, bounded by the GCN-fit cutoff year.
    user_max_year = history_interactions.groupby('user_idx')['interaction_year'].max().clip(upper=MAX_DATASET_YEAR).to_dict()

    dataset_rows = []
    # Only users who both appear in the graph (had >=k-core history edges) and
    # have at least one held-out label interaction can produce training rows.
    unique_users = [u for u in user_to_labels.keys() if u in user_to_history]

    print(f"Mining leakage-free hard negatives for {len(unique_users):,} users...")

    for u_idx in tqdm(unique_users):
        history_indices = user_to_history.get(u_idx, [])
        label_pos_indices = user_to_labels.get(u_idx, [])
        if not history_indices or not label_pos_indices:
            continue

        u_meta = user_metadata_dict[u_idx]
        u_count = u_meta['user_explicit_rating_count']
        ref_year = user_max_year.get(u_idx, MAX_DATASET_YEAR)

        # A. LightGCN Scores (embeddings never trained on label_pos_indices)
        u_vec_gcn = user_embs_gcn[u_idx]
        lightgcn_scores = np.dot(book_embs_gcn, u_vec_gcn)

        # B. Semantic Scores (profile built purely from pre-cutoff history)
        user_sem_profile = np.mean(aligned_sem_embs[history_indices], axis=0)
        profile_norm = np.linalg.norm(user_sem_profile)
        if profile_norm > 0:
            user_sem_profile /= profile_norm

        semantic_scores = np.dot(aligned_sem_embs, user_sem_profile)

        # C. Fuse Scores
        fused_scores, _ = fuse_scores(lightgcn_scores, semantic_scores, u_count)

        # D. Get Top K Candidate Indices, excluding books already in the user's
        # known history (mirrors real candidate generation, which never
        # re-recommends already-read books).
        candidate_scores = fused_scores.copy()
        candidate_scores[history_indices] = -np.inf
        top_k_indices = np.argpartition(candidate_scores, -STAGE_1_TOP_K)[-STAGE_1_TOP_K:]
        all_candidate_indices = set(top_k_indices).union(set(label_pos_indices))

        # E. Build Training Rows
        for b_idx in all_candidate_indices:
            is_positive = 1 if b_idx in label_pos_indices else 0

            dataset_rows.append({
                'user_idx': u_idx,
                'book_idx': b_idx,
                'target': is_positive,
                'lightgcn_score': lightgcn_scores[b_idx],
                'semantic_score': semantic_scores[b_idx],
                'fused_score': fused_scores[b_idx],
                'user_avg_rating': u_meta['user_avg_rating'],
                'user_explicit_rating_count': u_count,
                'reference_year': ref_year  # CONSISTENT REFERENCE YEAR FOR ALL CANDIDATES
            })

    print("Converting to DataFrame...")
    df_combined = pd.DataFrame(dataset_rows)
    
    print("Merging Book Metadata...")
    df_combined = df_combined.merge(book_map, on='book_idx')
    df_combined = df_combined.merge(books[['book_id', 'is_long_book', 'publication_year']], on='book_id', how='left')
    
    # --- LEAK FIX: Calculate age using consistent reference year for BOTH positives and negatives ---
    df_combined['book_age_at_interaction'] = np.maximum(
        0, df_combined['reference_year'] - df_combined['publication_year']
    )
    df_combined['book_age_at_interaction'].fillna(0, inplace=True)
    
    final_cols = [
        'user_idx', 'book_idx', 'target', 
        'lightgcn_score', 'semantic_score', 'fused_score',
        'user_avg_rating', 'user_explicit_rating_count', 
        'is_long_book', 'book_age_at_interaction'
    ]
    df_final = df_combined[final_cols]
    
    out_path = PROCESSED_DIR / "xgboost_dataset.parquet"
    df_final.to_parquet(out_path, index=False)
    
    print(f"\n✓ Leak-free dataset generated!")
    print(f"  Saved to: {out_path}")
    print(f"  Positives: {len(df_final[df_final['target'] == 1]):,} | Hard Negatives: {len(df_final[df_final['target'] == 0]):,}")

if __name__ == "__main__":
    build_hard_negative_dataset()