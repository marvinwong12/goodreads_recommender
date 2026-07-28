import numpy as np
import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(".").resolve() # adjust if needed
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
LIGHTGCN_DIR = PROJECT_ROOT / "models" / "lightgcn"

# Load Data
interactions = pd.read_parquet(PROCESSED_DIR / "interactions_clean.parquet")
book_map = pd.read_parquet(GRAPH_DIR / "book_mapping.parquet")
user_map = pd.read_parquet(GRAPH_DIR / "user_mapping.parquet")

user_embs_gcn = np.load(LIGHTGCN_DIR / "user_embeddings.npy")
book_embs_gcn = np.load(LIGHTGCN_DIR / "book_embeddings.npy")

# Test interactions
test_pos = interactions[(interactions['is_train'] == False) & (interactions['is_engagement'] == 1)]
test_pos = test_pos.merge(user_map, on='user_id').merge(book_map, on='book_id')

# Sample 1,000 test users
sample_users = test_pos['user_idx'].unique()[:1000]
user_to_test = test_pos.groupby('user_idx')['book_idx'].apply(set).to_dict()

stage1_hits = 0
total_users = 0

for u_idx in sample_users:
    gt_books = user_to_test[u_idx]
    
    # Pure LightGCN retrieval
    scores = np.dot(book_embs_gcn, user_embs_gcn[u_idx])
    top_100 = np.argpartition(scores, -100)[-100:]
    
    if any(b in gt_books for b in top_100):
        stage1_hits += 1
    total_users += 1

print(f"--> Pure LightGCN Stage 1 Recall@100: {stage1_hits / total_users:.4f}")