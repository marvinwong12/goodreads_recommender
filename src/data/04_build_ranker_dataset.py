# src/data/build_hard_negative_dataset.py
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Path Setup ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

# Add project root to sys.path so we can import from src.models
sys.path.append(str(PROJECT_ROOT))
from src.models.dynamic_fusion import fuse_scores 

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
LIGHTGCN_DIR = PROJECT_ROOT / "models" / "lightgcn"
VECTOR_DB_DIR = PROJECT_ROOT / "models" / "vector_index"

STAGE_1_TOP_K = 100

def build_hard_negative_dataset():
    print("Loading datasets and mappings...")
    interactions = pd.read_parquet(PROCESSED_DIR / "interactions_clean.parquet")
    books = pd.read_parquet(PROCESSED_DIR / "books_clean.parquet")
    book_map = pd.read_parquet(GRAPH_DIR / "book_mapping.parquet")
    user_map = pd.read_parquet(GRAPH_DIR / "user_mapping.parquet")
    
    num_books = len(book_map)
    
    # 1. Load LightGCN Embeddings
    print("Loading LightGCN Embeddings...")
    user_embs_gcn = np.load(LIGHTGCN_DIR / "user_embeddings.npy")
    book_embs_gcn = np.load(LIGHTGCN_DIR / "book_embeddings.npy")
    
    # 2. Load and Align Semantic Embeddings (Vector DB)
    print("Loading Semantic Vector DB...")
    sem_embs_raw = np.load(VECTOR_DB_DIR / "book_embeddings.npy")
    sem_ids_raw = np.load(VECTOR_DB_DIR / "book_ids.npy", allow_pickle=True)
    
    # We need to align the semantic vectors so their index matches the LightGCN `book_idx`
    sem_dim = sem_embs_raw.shape[1]
    aligned_sem_embs = np.zeros((num_books, sem_dim))
    
    # Create a quick lookup for book_id -> book_idx
    id_to_idx = dict(zip(book_map['book_id'], book_map['book_idx']))
    
    for i, b_id in enumerate(sem_ids_raw):
        # b_id might be a string or int depending on your DB, ensure it matches book_map
        b_id_val = int(b_id) if str(b_id).isdigit() else b_id 
        if b_id_val in id_to_idx:
            b_idx = id_to_idx[b_id_val]
            aligned_sem_embs[b_idx] = sem_embs_raw[i]
            
    # Normalize semantic vectors for fast Cosine Similarity (dot product of normalized vectors)
    norms = np.linalg.norm(aligned_sem_embs, axis=1, keepdims=True)
    aligned_sem_embs = np.divide(aligned_sem_embs, norms, out=np.zeros_like(aligned_sem_embs), where=norms!=0)
    
    # 3. Prepare User Interaction Data
    train_interactions = interactions[interactions['is_train'] == True]
    positives = train_interactions[train_interactions['is_engagement'] == 1].copy()
    positives = positives.merge(user_map, on='user_id').merge(book_map, on='book_id')
    
    user_metadata = positives[['user_idx', 'user_avg_rating', 'user_explicit_rating_count']].drop_duplicates().set_index('user_idx')
    user_to_positives = positives.groupby('user_idx')['book_idx'].apply(list).to_dict()
    pos_year_map = positives.set_index(['user_idx', 'book_idx'])['interaction_year'].to_dict()
    
    dataset_rows = []
    unique_users = positives['user_idx'].unique()
    
    print(f"Simulating Stage 1 & Mining Hard Negatives for {len(unique_users):,} users...")
    
    for u_idx in tqdm(unique_users):
        true_pos_indices = user_to_positives.get(u_idx, [])
        if not true_pos_indices:
            continue
            
        # A. Get LightGCN Scores for ALL books
        u_vec_gcn = user_embs_gcn[u_idx]
        lightgcn_scores = np.dot(book_embs_gcn, u_vec_gcn)
        
        # B. Get Semantic Scores for ALL books
        # Build the user's semantic profile by averaging the vectors of books they read
        user_sem_profile = np.mean(aligned_sem_embs[true_pos_indices], axis=0)
        profile_norm = np.linalg.norm(user_sem_profile)
        if profile_norm > 0:
            user_sem_profile = user_sem_profile / profile_norm
            
        semantic_scores = np.dot(aligned_sem_embs, user_sem_profile)
        
        # C. Apply Dynamic Fusion
        u_count = user_metadata.loc[u_idx, 'user_explicit_rating_count']
        
        # Using the fuse_scores function you built!
        fused_scores, alpha_used = fuse_scores(
            lightgcn_scores=lightgcn_scores,
            semantic_scores=semantic_scores,
            user_interaction_count=u_count
        )
        
        # D. Pull Top K Candidates
        top_k_indices = np.argsort(fused_scores)[::-1][:STAGE_1_TOP_K]
        all_candidate_indices = set(top_k_indices).union(set(true_pos_indices))
        
        # E. Build Training Rows
        for b_idx in all_candidate_indices:
            is_positive = 1 if b_idx in true_pos_indices else 0
            int_year = pos_year_map.get((u_idx, b_idx), 2022) 
            
            dataset_rows.append({
                'user_idx': u_idx,
                'book_idx': b_idx,
                'target': is_positive,
                'lightgcn_score': lightgcn_scores[b_idx],
                'semantic_score': semantic_scores[b_idx],
                'fused_score': fused_scores[b_idx], 
                'user_avg_rating': user_metadata.loc[u_idx, 'user_avg_rating'],
                'user_explicit_rating_count': u_count,
                'interaction_year': int_year
            })

    print("Converting to DataFrame...")
    df_combined = pd.DataFrame(dataset_rows)
    
    print("Merging Book Metadata...")
    df_combined = df_combined.merge(book_map, on='book_idx')
    df_combined = df_combined.merge(
        books[['book_id', 'is_long_book', 'publication_year']], 
        on='book_id', 
        how='left'
    )
    
    df_combined['book_age_at_interaction'] = np.maximum(
        0, df_combined['interaction_year'] - df_combined['publication_year']
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
    
    num_pos = len(df_final[df_final['target'] == 1])
    num_neg = len(df_final[df_final['target'] == 0])
    print(f"\n✓ Dataset ready with REAL semantic embeddings!")
    print(f"  Saved to: {out_path}")
    print(f"  Positives: {num_pos:,} | Hard Negatives: {num_neg:,}")

if __name__ == "__main__":
    build_hard_negative_dataset()