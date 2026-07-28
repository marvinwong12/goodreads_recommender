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
    train_interactions = interactions[interactions['is_train'] == True]
    positives = train_interactions[train_interactions['is_engagement'] == 1].copy()
    positives = positives.merge(user_map, on='user_id').merge(book_map, on='book_id')
    
    user_metadata_dict = positives[['user_idx', 'user_avg_rating', 'user_explicit_rating_count']].drop_duplicates().set_index('user_idx').to_dict('index')
    user_to_positives = positives.groupby('user_idx')['book_idx'].apply(list).to_dict()
    
    # Map each user to their most recent interaction year (bounded by 2017)
    user_max_year = positives.groupby('user_idx')['interaction_year'].max().clip(upper=MAX_DATASET_YEAR).to_dict()
    
    dataset_rows = []
    unique_users = list(user_to_positives.keys())
    
    print(f"Mining leakage-free hard negatives for {len(unique_users):,} users...")
    
    for u_idx in tqdm(unique_users):
        true_pos_indices = user_to_positives.get(u_idx, [])
        if not true_pos_indices:
            continue
            
        u_meta = user_metadata_dict[u_idx]
        u_count = u_meta['user_explicit_rating_count']
        ref_year = user_max_year.get(u_idx, MAX_DATASET_YEAR)
        
        # A. LightGCN Scores
        u_vec_gcn = user_embs_gcn[u_idx]
        lightgcn_scores = np.dot(book_embs_gcn, u_vec_gcn)
        
        # B. Semantic Scores
        user_sem_profile = np.mean(aligned_sem_embs[true_pos_indices], axis=0)
        profile_norm = np.linalg.norm(user_sem_profile)
        if profile_norm > 0:
            user_sem_profile /= profile_norm
            
        semantic_scores = np.dot(aligned_sem_embs, user_sem_profile)
        
        # C. Fuse Scores
        fused_scores, _ = fuse_scores(lightgcn_scores, semantic_scores, u_count)
        
        # D. Get Top K Candidate Indices
        top_k_indices = np.argpartition(fused_scores, -STAGE_1_TOP_K)[-STAGE_1_TOP_K:]
        all_candidate_indices = set(top_k_indices).union(set(true_pos_indices))
        
        num_pos = len(true_pos_indices)
        
        # E. Build Training Rows
        for b_idx in all_candidate_indices:
            is_positive = 1 if b_idx in true_pos_indices else 0
            
            # --- LEAK FIX: Leave-One-Out adjustment for positive item semantic score ---
            b_sem_score = semantic_scores[b_idx]
            b_fused_score = fused_scores[b_idx]
            
            if is_positive and num_pos > 1:
                # Remove self-vector contribution to prevent self-similarity leakage
                adjusted_profile = (user_sem_profile * num_pos - aligned_sem_embs[b_idx]) / (num_pos - 1)
                adj_norm = np.linalg.norm(adjusted_profile)
                if adj_norm > 0:
                    adjusted_profile /= adj_norm
                b_sem_score = np.dot(aligned_sem_embs[b_idx], adjusted_profile)
                
                # Re-fuse with adjusted semantic score
                temp_fused, _ = fuse_scores(np.array([lightgcn_scores[b_idx]]), np.array([b_sem_score]), u_count)
                b_fused_score = temp_fused[0]
            
            dataset_rows.append({
                'user_idx': u_idx,
                'book_idx': b_idx,
                'target': is_positive,
                'lightgcn_score': lightgcn_scores[b_idx],
                'semantic_score': b_sem_score,
                'fused_score': b_fused_score, 
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