import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
sys.path.append(str(PROJECT_ROOT))

from src.models.dynamic_fusion import fuse_scores

# Paths
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
LIGHTGCN_DIR = PROJECT_ROOT / "models" / "lightgcn"
VECTOR_DB_DIR = PROJECT_ROOT / "models" / "vector_index"
XGBOOST_DIR = PROJECT_ROOT / "models" / "xgboost"

print("Loading data for pipeline diagnostic...")
interactions = pd.read_parquet(PROCESSED_DIR / "interactions_clean.parquet")
books = pd.read_parquet(PROCESSED_DIR / "books_clean.parquet")
book_map = pd.read_parquet(GRAPH_DIR / "book_mapping.parquet")
user_map = pd.read_parquet(GRAPH_DIR / "user_mapping.parquet")

num_books = len(book_map)

# Embeddings
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

# XGBoost
ranker = xgb.XGBRanker()
ranker.load_model(XGBOOST_DIR / "ranker_model.json")

# Lookups
train_pos = interactions[(interactions['is_train'] == True) & (interactions['is_engagement'] == 1)].merge(user_map, on='user_id').merge(book_map, on='book_id')
test_pos = interactions[(interactions['is_train'] == False) & (interactions['is_engagement'] == 1)].merge(user_map, on='user_id').merge(book_map, on='book_id')

user_to_train_pos = train_pos.groupby('user_idx')['book_idx'].apply(list).to_dict()
user_to_test_pos = test_pos.groupby('user_idx')['book_idx'].apply(set).to_dict()
user_metadata_dict = train_pos[['user_idx', 'user_avg_rating', 'user_explicit_rating_count']].drop_duplicates().set_index('user_idx').to_dict('index')

# Feature pre-allocations
book_meta = book_map.merge(books[['book_id', 'is_long_book', 'publication_year']], on='book_id', how='left')
is_long_arr = np.zeros(num_books)
pub_year_arr = np.full(num_books, 2017)
for _, row in book_meta.iterrows():
    b_idx = int(row['book_idx'])
    is_long_arr[b_idx] = row['is_long_book'] if pd.notnull(row['is_long_book']) else 0
    pub_year_arr[b_idx] = row['publication_year'] if pd.notnull(row['publication_year']) else 2020
book_age_arr = np.maximum(0, 2017 - pub_year_arr)

sample_users = [u for u in list(user_to_test_pos.keys()) if u in user_to_train_pos][:1000]

l_recall100, f_recall100 = 0, 0
l_hr10, f_hr10, xgb_hr10 = 0, 0, 0

feature_names = ['lightgcn_score', 'semantic_score', 'fused_score', 'user_avg_rating', 'user_explicit_rating_count', 'is_long_book', 'book_age_at_interaction']

for u_idx in sample_users:
    gt_set = user_to_test_pos[u_idx]
    train_indices = user_to_train_pos[u_idx]
    u_meta = user_metadata_dict.get(u_idx, {'user_avg_rating': 3.5, 'user_explicit_rating_count': 1})
    
    # 1. LightGCN Scores
    lg_scores = np.dot(book_embs_gcn, user_embs_gcn[u_idx])
    
    lg_top100 = np.argpartition(lg_scores, -100)[-100:]
    lg_top10 = lg_top100[np.argsort(lg_scores[lg_top100])[::-1][:10]]
    if any(b in gt_set for b in lg_top100): l_recall100 += 1
    if any(b in gt_set for b in lg_top10): l_hr10 += 1
    
    # 2. Fused Scores
    u_sem_profile = np.mean(aligned_sem_embs[train_indices], axis=0)
    p_norm = np.linalg.norm(u_sem_profile)
    if p_norm > 0: u_sem_profile /= p_norm
    sem_scores = np.dot(aligned_sem_embs, u_sem_profile)
    
    fused_scores, _ = fuse_scores(lg_scores, sem_scores, u_meta['user_explicit_rating_count'])
    fused_scores[train_indices] = -np.inf
    lg_scores[train_indices] = -np.inf
    sem_scores[train_indices] = -np.inf
    
    f_top100 = np.argpartition(fused_scores, -100)[-100:]
    f_top10 = f_top100[np.argsort(fused_scores[f_top100])[::-1][:10]]
    if any(b in gt_set for b in f_top100): f_recall100 += 1
    if any(b in gt_set for b in f_top10): f_hr10 += 1
    
    # 3. XGBoost Re-ranking on Fused Top 100
    c_np = np.column_stack((
        lg_scores[f_top100], sem_scores[f_top100], fused_scores[f_top100],
        np.full(100, u_meta['user_avg_rating']), np.full(100, u_meta['user_explicit_rating_count']),
        is_long_arr[f_top100], book_age_arr[f_top100]
    ))
    X_cand = pd.DataFrame(c_np, columns=feature_names)
    scores = ranker.predict(X_cand)

    xgb_top10 = f_top100[np.argsort(scores)[::-1][:10]]
    if any(b in gt_set for b in xgb_top10): xgb_hr10 += 1

N = len(sample_users)
print("\n================ PIPELINE DIAGNOSTIC ================")
print(f"1. Pure LightGCN   --> Recall@100: {l_recall100/N:.4f} | HR@10: {l_hr10/N:.4f}")
print(f"2. Fused Scores    --> Recall@100: {f_recall100/N:.4f} | HR@10: {f_hr10/N:.4f}")
print(f"3. XGBoost Ranked  --> HR@10: {xgb_hr10/N:.4f}")
print("=====================================================")


import xgboost as xgb
ranker = xgb.XGBRanker()
ranker.load_model("models/xgboost/ranker_model.json")
print("Expected XGBoost Features:", ranker.get_booster().feature_names)