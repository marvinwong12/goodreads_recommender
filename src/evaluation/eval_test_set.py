import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
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
XGBOOST_DIR = PROJECT_ROOT / "models" / "xgboost"
RESULTS_DIR = PROJECT_ROOT / "results"

STAGE_1_TOP_K = 100
INFERENCE_YEAR = 2024  # Used to calculate book age during live inference

def compute_ndcg_at_k(actual_hits, k):
    if not actual_hits:
        return 0.0
    
    dcg = 0.0
    for rank, hit in enumerate(actual_hits[:k], start=1):
        if hit:
            dcg += 1.0 / np.log2(rank + 1)
            
    ideal_hits_count = min(sum(actual_hits), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits_count + 1))
    
    return dcg / idcg if idcg > 0 else 0.0

def evaluate():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading test datasets and artifacts...")
    
    # 1. Load Data
    interactions = pd.read_parquet(PROCESSED_DIR / "interactions_clean.parquet")
    books = pd.read_parquet(PROCESSED_DIR / "books_clean.parquet")
    book_map = pd.read_parquet(GRAPH_DIR / "book_mapping.parquet")
    user_map = pd.read_parquet(GRAPH_DIR / "user_mapping.parquet")
    
    num_books = len(book_map)
    
    # 2. Load Embeddings
    user_embs_gcn = np.load(LIGHTGCN_DIR / "user_embeddings.npy")
    book_embs_gcn = np.load(LIGHTGCN_DIR / "book_embeddings.npy")
    
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
    
    # 3. Load Trained Ranker
    ranker = xgb.XGBRanker()
    ranker.load_model(XGBOOST_DIR / "ranker_model.json")
    
    # 4. Prepare Evaluation Lookups
    train_interactions = interactions[interactions['is_train'] == True]
    test_interactions = interactions[(interactions['is_train'] == False) & (interactions['is_engagement'] == 1)]
    
    train_positives = train_interactions[train_interactions['is_engagement'] == 1].merge(user_map, on='user_id').merge(book_map, on='book_id')
    test_positives = test_interactions.merge(user_map, on='user_id').merge(book_map, on='book_id')
    
    user_metadata_dict = train_positives[['user_idx', 'user_avg_rating', 'user_explicit_rating_count']].drop_duplicates().set_index('user_idx').to_dict('index')
    user_to_train_pos = train_positives.groupby('user_idx')['book_idx'].apply(list).to_dict()
    user_to_test_pos = test_positives.groupby('user_idx')['book_idx'].apply(set).to_dict()
    
    # Static book feature arrays
    book_meta = book_map.merge(books[['book_id', 'is_long_book', 'publication_year']], on='book_id', how='left')
    
    is_long_arr = np.zeros(num_books)
    pub_year_arr = np.full(num_books, INFERENCE_YEAR)
    
    for _, row in book_meta.iterrows():
        b_idx = int(row['book_idx'])
        is_long_arr[b_idx] = row['is_long_book'] if pd.notnull(row['is_long_book']) else 0
        pub_year_arr[b_idx] = row['publication_year'] if pd.notnull(row['publication_year']) else 2020
        
    book_age_arr = np.maximum(0, INFERENCE_YEAR - pub_year_arr)
    
    test_users = list(user_to_test_pos.keys())
    print(f"Evaluating {len(test_users):,} test users...")
    
    metrics = {'hr@5': [], 'hr@10': [], 'ndcg@5': [], 'ndcg@10': [], 'mrr@10': []}
    
    feature_names = [
        'lightgcn_score', 'semantic_score', 'fused_score', 
        'user_avg_rating', 'user_explicit_rating_count', 
        'is_long_book', 'book_age_at_interaction'
    ]
    
    for u_idx in tqdm(test_users):
        train_pos_indices = user_to_train_pos.get(u_idx, [])
        ground_truth_set = user_to_test_pos[u_idx]
        
        if not train_pos_indices or not ground_truth_set:
            continue
            
        u_meta = user_metadata_dict.get(u_idx, {'user_avg_rating': 3.5, 'user_explicit_rating_count': 1})
        u_count = u_meta['user_explicit_rating_count']
        
        # --- STAGE 1: Candidate Generation ---
        u_vec_gcn = user_embs_gcn[u_idx]
        lightgcn_scores = np.dot(book_embs_gcn, u_vec_gcn)
        
        user_sem_profile = np.mean(aligned_sem_embs[train_pos_indices], axis=0)
        profile_norm = np.linalg.norm(user_sem_profile)
        if profile_norm > 0:
            user_sem_profile = user_sem_profile / profile_norm
        semantic_scores = np.dot(aligned_sem_embs, user_sem_profile)
        
        # 1. FUSE RAW SCORES FIRST (Prevents NaNs during min-max scaling)
        fused_scores, _ = fuse_scores(lightgcn_scores, semantic_scores, u_count)
        
        # 2. MASK TRAINED HISTORY AFTER FUSION
        lightgcn_scores[train_pos_indices] = -np.inf
        semantic_scores[train_pos_indices] = -np.inf
        fused_scores[train_pos_indices] = -np.inf
        
        # 3. Retrieve Top K candidates
        candidate_indices = np.argpartition(fused_scores, -STAGE_1_TOP_K)[-STAGE_1_TOP_K:]
        
        # --- STAGE 2: Vectorized XGBoost Re-ranking ---
        c_lightgcn = lightgcn_scores[candidate_indices]
        c_semantic = semantic_scores[candidate_indices]
        c_fused = fused_scores[candidate_indices]
        c_u_avg = np.full(STAGE_1_TOP_K, u_meta['user_avg_rating'])
        c_u_cnt = np.full(STAGE_1_TOP_K, u_count)
        c_is_long = is_long_arr[candidate_indices]
        c_age = book_age_arr[candidate_indices]
        
        X_candidates_np = np.column_stack((
            c_lightgcn, c_semantic, c_fused, c_u_avg, c_u_cnt, c_is_long, c_age
        ))
        
        X_candidates = pd.DataFrame(X_candidates_np, columns=feature_names)
        ranker_scores = ranker.predict(X_candidates)

        # Sort by ranker scores (Descending)
        reranked_order = np.argsort(ranker_scores)[::-1]
        final_ranked_book_indices = candidate_indices[reranked_order]
        
        # --- METRIC COMPUTATION ---
        hits_binary = [b_idx in ground_truth_set for b_idx in final_ranked_book_indices]
        
        metrics['hr@5'].append(1.0 if any(hits_binary[:5]) else 0.0)
        metrics['hr@10'].append(1.0 if any(hits_binary[:10]) else 0.0)
        metrics['ndcg@5'].append(compute_ndcg_at_k(hits_binary, 5))
        metrics['ndcg@10'].append(compute_ndcg_at_k(hits_binary, 10))
        
        mrr = 0.0
        for rank, hit in enumerate(hits_binary[:10], start=1):
            if hit:
                mrr = 1.0 / rank
                break
        metrics['mrr@10'].append(mrr)

    print("\n==============================================")
    print("      FINAL TWO-STAGE PIPELINE EVALUATION     ")
    print("==============================================")
    final_summary = {}
    for metric_name, values in metrics.items():
        avg_val = np.mean(values)
        final_summary[metric_name] = round(float(avg_val), 4)
        print(f"  {metric_name.upper():<10}: {avg_val:.4f}")
    print("==============================================")
    
    out_file = RESULTS_DIR / "test_evaluation_results.json"
    with open(out_file, "w") as f:
        json.dump(final_summary, f, indent=4)
    print(f"✓ Results exported to: {out_file}")

if __name__ == "__main__":
    evaluate()