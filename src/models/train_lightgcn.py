# src/models/train_lightgcn.py
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
LIGHTGCN_DIR = PROCESSED_DIR / "lightgcn"

# Hyperparameters
EMBEDDING_DIM = 64
NUM_LAYERS = 3
BATCH_SIZE = 16384
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Dataset for BPR Triplet Sampling (User, Pos_Item, Neg_Item)
class BPRDataset(Dataset):
    def __init__(self, edges_df, num_users, num_books):
        self.num_users = num_users
        self.num_books = num_books
        
        # User -> Set of positive books for fast negative sampling lookup
        self.user_to_pos = edges_df.groupby("user_idx")["book_idx"].apply(set).to_dict()
        self.edges = edges_df[["user_idx", "book_idx"]].values
        
    def __len__(self):
        return len(self.edges)
        
    def __getitem__(self, idx):
        u, pos_i = self.edges[idx]
        
        # Sample negative item j that user u has NOT interacted with
        pos_set = self.user_to_pos.get(u, set())
        neg_j = np.random.randint(0, self.num_books)
        while neg_j in pos_set:
            neg_j = np.random.randint(0, self.num_books)
            
        return u, pos_i, neg_j

# PyTorch LightGCN Module
class LightGCN(nn.Module):
    def __init__(self, num_users, num_books, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_books = num_books
        self.num_layers = num_layers
        
        # Trainable initial node embeddings E_0
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)
        
        # Xavier Normal initialization
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.book_embedding.weight, std=0.1)
        
    def compute_norm_adj(self, edge_index):
        """Builds symmetrically normalized bipartite adjacency matrix A~"""
        user_nodes = edge_index[0]
        book_nodes = edge_index[1] + self.num_users  # Shift book indices for bipartite graph
        
        # Undirected graph edges
        row = torch.cat([user_nodes, book_nodes])
        col = torch.cat([book_nodes, user_nodes])
        
        total_nodes = self.num_users + self.num_books
        
        # Compute node degrees
        deg = torch.bincount(row, minlength=total_nodes).float()
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        # Edge weights d_i^(-1/2) * d_j^(-1/2)
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        indices = torch.stack([row, col])
        norm_adj = torch.sparse_coo_tensor(indices, edge_weight, torch.Size([total_nodes, total_nodes]))
        return norm_adj.coalesce()

    def forward(self, norm_adj):
        # Initial embeddings layer E_0
        ego_embeddings = torch.cat([self.user_embedding.weight, self.book_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        # Graph convolution across K layers
        for layer in range(self.num_layers):
            ego_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        # Final embedding is uniform average across all layer outputs
        final_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        
        user_final = final_embeddings[:self.num_users]
        book_final = final_embeddings[self.num_users:]
        
        return user_final, book_final

def train():
    LIGHTGCN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using Compute Device: {DEVICE}")
    
    # Load graph metadata and edges
    with open(GRAPH_DIR / "graph_meta.json", "r") as f:
        meta = json.load(f)
        
    num_users = meta["num_users"]
    num_books = meta["num_books"]
    
    print(f"Loading K-Core Graph ({meta['num_edges']:,} edges, {num_users:,} users, {num_books:,} books)...")
    edges_df = pd.read_parquet(GRAPH_DIR / f"edges_k{meta['k_core']}.parquet")
    
    # Build dataset and dataloader
    dataset = BPRDataset(edges_df, num_users, num_books)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Initialize LightGCN Model
    model = LightGCN(num_users, num_books, embedding_dim=EMBEDDING_DIM, num_layers=NUM_LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Pre-compute sparse normalized adjacency matrix
    edge_index = torch.tensor(edges_df[["user_idx", "book_idx"]].values.T, dtype=torch.long).to(DEVICE)
    print("Building normalized bipartite adjacency matrix...")
    norm_adj = model.compute_norm_adj(edge_index).to(DEVICE)
    print("✓ Graph normalized successfully!")
    
    print(f"\nStarting LightGCN Training ({EPOCHS} Epochs)...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        # Compute forward pass graph propagation once per epoch
        user_final, book_final = model(norm_adj)
        
        for u, pos_i, neg_j in tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}"):
            u = u.to(DEVICE)
            pos_i = pos_i.to(DEVICE)
            neg_j = neg_j.to(DEVICE)
            
            u_emb = user_final[u]
            pos_emb = book_final[pos_i]
            neg_emb = book_final[neg_j]
            
            # Initial embeddings for L2 regularization
            u_0 = model.user_embedding(u)
            pos_0 = model.book_embedding(pos_i)
            neg_0 = model.book_embedding(neg_j)
            
            # Predict scores
            pos_scores = (u_emb * pos_emb).sum(dim=1)
            neg_scores = (u_emb * neg_emb).sum(dim=1)
            
            # BPR Loss: -ln(sigmoid(pos_score - neg_score))
            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
            
            # L2 Regularization Loss
            reg_loss = WEIGHT_DECAY * (u_0.norm(2).pow(2) + pos_0.norm(2).pow(2) + neg_0.norm(2).pow(2)) / BATCH_SIZE
            
            loss = bpr_loss + reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        elapsed = time.time() - start_time
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Loss: {avg_loss:.6f} | Time: {elapsed:.2f}s")
        
    # Extract & Save Final Learned Embeddings
    print("\nSaving final user and book embedding vectors...")
    model.eval()
    with torch.no_grad():
        final_u, final_b = model(norm_adj)
        
    np.save(LIGHTGCN_DIR / "user_embeddings.npy", final_u.cpu().numpy())
    np.save(LIGHTGCN_DIR / "book_embeddings.npy", final_b.cpu().numpy())
    torch.save(model.state_dict(), LIGHTGCN_DIR / "lightgcn_model.pt")
    
    print("==========================================")
    print("✓ LightGCN Training Complete!")
    print(f"  Saved User Embeddings : {LIGHTGCN_DIR / 'user_embeddings.npy'}")
    print(f"  Saved Book Embeddings : {LIGHTGCN_DIR / 'book_embeddings.npy'}")
    print("==========================================")

if __name__ == "__main__":
    train()