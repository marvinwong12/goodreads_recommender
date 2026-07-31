# src/models/train_lightgcn.py
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
MODEL_DIR = PROJECT_ROOT / "models"
LIGHTGCN_DIR = MODEL_DIR / "lightgcn"

# Hyperparameters
EMBEDDING_DIM = 64
NUM_LAYERS = 3
BATCH_SIZE = 32768  # KEEP BIG
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Negative sampling: multiple negatives per positive, drawn proportional to
# book_degree^NEG_POWER (word2vec-style) rather than uniformly. Uniform
# negatives get "too easy" as training progresses (a random book is almost
# always an obvious non-match), so the BPR gradient vanishes early. Sampling
# harder (popular-but-still-negative) candidates keeps the signal useful.
NEG_PER_POS = 4
NEG_POWER = 0.75

VAL_USERS = 5000
VAL_K = 100
VAL_CHUNK_SIZE = 500

class LightGCN(nn.Module):
    def __init__(self, num_users, num_books, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_books = num_books
        self.num_layers = num_layers
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)
        
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.book_embedding.weight, std=0.1)
        
    def compute_norm_adj(self, edge_index):
        user_nodes = edge_index[0]
        book_nodes = edge_index[1] + self.num_users
        
        row = torch.cat([user_nodes, book_nodes])
        col = torch.cat([book_nodes, user_nodes])
        
        total_nodes = self.num_users + self.num_books
        
        deg = torch.bincount(row, minlength=total_nodes).float()
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        indices = torch.stack([row, col])
        norm_adj = torch.sparse_coo_tensor(indices, edge_weight, torch.Size([total_nodes, total_nodes]))
        return norm_adj.coalesce()

    def forward(self, norm_adj):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.book_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for layer in range(self.num_layers):
            # Sparse matrix multiplication propagates embeddings across the graph
            ego_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        final_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        
        user_final = final_embeddings[:self.num_users]
        book_final = final_embeddings[self.num_users:]
        
        return user_final, book_final

def build_validation_data(seed=42):
    """
    Held-out recall@K signal, reusing the same leakage-free split as the
    ranker dataset: `history` (is_gcn_fit, what LightGCN's edges are built
    from) vs. `label` (is_train & ~is_gcn_fit, never seen during graph
    training). A user's label books are checked for appearance in the
    top-K of their scores after masking out their history books.
    """
    interactions = pd.read_parquet(PROCESSED_DIR / "interactions_clean.parquet")
    user_map = pd.read_parquet(GRAPH_DIR / "user_mapping.parquet")
    book_map = pd.read_parquet(GRAPH_DIR / "book_mapping.parquet")

    train_interactions = interactions[interactions["is_train"] == True]
    history_interactions = train_interactions[
        (train_interactions["is_gcn_fit"] == True) & (train_interactions["is_engagement"] == 1)
    ].merge(user_map, on="user_id").merge(book_map, on="book_id")
    label_interactions = train_interactions[
        (train_interactions["is_gcn_fit"] == False) & (train_interactions["is_engagement"] == 1)
    ].merge(user_map, on="user_id").merge(book_map, on="book_id")

    user_to_history = history_interactions.groupby("user_idx")["book_idx"].apply(list).to_dict()
    user_to_label = label_interactions.groupby("user_idx")["book_idx"].apply(list).to_dict()

    val_users = [u for u in user_to_label if u in user_to_history]
    rng = np.random.default_rng(seed)
    if len(val_users) > VAL_USERS:
        val_users = rng.choice(val_users, size=VAL_USERS, replace=False).tolist()

    return val_users, user_to_history, user_to_label


@torch.no_grad()
def evaluate_recall_at_k(user_final, book_final, val_users, user_to_history, user_to_label, k=VAL_K):
    hits = 0
    total = 0
    for start in range(0, len(val_users), VAL_CHUNK_SIZE):
        chunk = val_users[start:start + VAL_CHUNK_SIZE]
        u_emb = user_final[torch.tensor(chunk, device=user_final.device)]
        scores = u_emb @ book_final.T  # (chunk, num_books)

        for i, u in enumerate(chunk):
            hist = user_to_history.get(u, [])
            if hist:
                scores[i, hist] = -float("inf")

        topk = torch.topk(scores, k, dim=1).indices.cpu().numpy()
        for i, u in enumerate(chunk):
            labels = user_to_label.get(u, [])
            if not labels:
                continue
            hits += len(set(topk[i].tolist()) & set(labels))
            total += min(len(labels), k)

    return hits / total if total > 0 else 0.0


def train():
    LIGHTGCN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using Compute Device: {DEVICE}")
    
    with open(GRAPH_DIR / "graph_meta.json", "r") as f:
        meta = json.load(f)
        
    num_users = meta["num_users"]
    num_books = meta["num_books"]
    
    print(f"Loading K-Core Graph ({meta['num_edges']:,} edges, {num_users:,} users, {num_books:,} books)...")
    edges_df = pd.read_parquet(GRAPH_DIR / f"edges_k{meta['k_core']}.parquet")
    
    # Push all edges directly to the GPU for extremely fast slicing
    print("Moving dataset to VRAM...")
    edges_tensor = torch.tensor(edges_df[["user_idx", "book_idx"]].values, dtype=torch.long, device=DEVICE)
    
    model = LightGCN(num_users, num_books, embedding_dim=EMBEDDING_DIM, num_layers=NUM_LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Building normalized bipartite adjacency matrix...")
    edge_index = edges_tensor.T
    norm_adj = model.compute_norm_adj(edge_index).to(DEVICE)
    print("✓ Graph normalized successfully!")

    print("Building popularity-weighted negative sampling distribution...")
    book_degree = torch.bincount(edges_tensor[:, 1], minlength=num_books).float()
    neg_sampling_probs = book_degree.pow(NEG_POWER)
    neg_sampling_probs = neg_sampling_probs / neg_sampling_probs.sum()

    print("Loading held-out validation split...")
    val_users, user_to_history, user_to_label = build_validation_data()
    print(f"✓ Validating recall@{VAL_K} on {len(val_users):,} held-out users")

    best_recall = -1.0

    print(f"\nStarting LightGCN Training ({EPOCHS} Epochs)...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        # Shuffle edges per epoch fully on the GPU
        perm = torch.randperm(len(edges_tensor), device=DEVICE)
        shuffled_edges = edges_tensor[perm]
        
        num_batches = len(shuffled_edges) // BATCH_SIZE + (1 if len(shuffled_edges) % BATCH_SIZE != 0 else 0)
        
        with tqdm(total=num_batches, desc=f"Epoch {epoch}/{EPOCHS}") as pbar:
            for i in range(0, len(shuffled_edges), BATCH_SIZE):
                batch = shuffled_edges[i:i + BATCH_SIZE]
                u = batch[:, 0]
                pos_i = batch[:, 1]

                # Multiple negatives per positive, sampled proportional to
                # book_degree^NEG_POWER rather than uniformly (harder than a
                # plain random book, which becomes a trivial negative early on).
                neg_j = torch.multinomial(
                    neg_sampling_probs, len(batch) * NEG_PER_POS, replacement=True
                ).view(len(batch), NEG_PER_POS)

                # Forward pass across full graph graph per batch
                user_final, book_final = model(norm_adj)

                u_emb = user_final[u]
                pos_emb = book_final[pos_i]
                neg_emb = book_final[neg_j]  # (batch, NEG_PER_POS, dim)

                u_0 = model.user_embedding(u)
                pos_0 = model.book_embedding(pos_i)
                neg_0 = model.book_embedding(neg_j)

                pos_scores = (u_emb * pos_emb).sum(dim=1)
                neg_scores = (u_emb.unsqueeze(1) * neg_emb).sum(dim=2)  # (batch, NEG_PER_POS)

                bpr_loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()
                reg_loss = (1/2) * WEIGHT_DECAY * (u_0.norm(2).pow(2) + pos_0.norm(2).pow(2) + neg_0.norm(2).pow(2)) / len(batch)
                
                loss = bpr_loss + reg_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.update(1)
            
        elapsed = time.time() - start_time
        avg_loss = total_loss / num_batches

        model.eval()
        with torch.no_grad():
            final_u, final_b = model(norm_adj)
        recall = evaluate_recall_at_k(final_u, final_b, val_users, user_to_history, user_to_label)

        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Loss: {avg_loss:.6f} | "
              f"Val Recall@{VAL_K}: {recall:.4f} | Time: {elapsed:.2f}s")

        if recall > best_recall:
            best_recall = recall
            np.save(LIGHTGCN_DIR / "user_embeddings.npy", final_u.cpu().numpy())
            np.save(LIGHTGCN_DIR / "book_embeddings.npy", final_b.cpu().numpy())
            torch.save(model.state_dict(), LIGHTGCN_DIR / "lightgcn_model.pt")
            print(f"  ✓ New best (Recall@{VAL_K}: {best_recall:.4f}) — checkpoint saved.")

    print("==========================================")
    print("✓ LightGCN Training Complete!")
    print(f"  Best Val Recall@{VAL_K}: {best_recall:.4f}")
    print(f"  Saved to : {LIGHTGCN_DIR}")
    print("==========================================")

if __name__ == "__main__":
    train()