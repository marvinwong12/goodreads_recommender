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

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
MODEL_DIR = PROJECT_ROOT / "models"
LIGHTGCN_DIR = MODEL_DIR / "lightgcn"

# Hyperparameters
EMBEDDING_DIM = 64
NUM_LAYERS = 3
BATCH_SIZE = 32768
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

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
                
                # Fast Vectorized Negative Sampling 
                # (With 99.94% sparsity, a random sample is a true negative 99.94% of the time)
                neg_j = torch.randint(0, num_books, (len(batch),), device=DEVICE)
                
                # Forward pass across full graph graph per batch
                user_final, book_final = model(norm_adj)
                
                u_emb = user_final[u]
                pos_emb = book_final[pos_i]
                neg_emb = book_final[neg_j]
                
                u_0 = model.user_embedding(u)
                pos_0 = model.book_embedding(pos_i)
                neg_0 = model.book_embedding(neg_j)
                
                pos_scores = (u_emb * pos_emb).sum(dim=1)
                neg_scores = (u_emb * neg_emb).sum(dim=1)
                
                bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
                reg_loss = WEIGHT_DECAY * (u_0.norm(2).pow(2) + pos_0.norm(2).pow(2) + neg_0.norm(2).pow(2)) / len(batch)
                
                loss = bpr_loss + reg_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.update(1)
            
        elapsed = time.time() - start_time
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Loss: {avg_loss:.6f} | Time: {elapsed:.2f}s")
        
    print("\nSaving final user and book embedding vectors to models/lightgcn/...")
    model.eval()
    with torch.no_grad():
        final_u, final_b = model(norm_adj)
        
    np.save(LIGHTGCN_DIR / "user_embeddings.npy", final_u.cpu().numpy())
    np.save(LIGHTGCN_DIR / "book_embeddings.npy", final_b.cpu().numpy())
    torch.save(model.state_dict(), LIGHTGCN_DIR / "lightgcn_model.pt")
    
    print("==========================================")
    print("✓ LightGCN Training Complete!")
    print(f"  Saved to : {LIGHTGCN_DIR}")
    print("==========================================")

if __name__ == "__main__":
    train()