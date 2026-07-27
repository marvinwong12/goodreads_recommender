import numpy as np
import pandas as pd
from pathlib import Path

# --- Setup Paths ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
LIGHTGCN_DIR = PROJECT_ROOT / "models" / "lightgcn"

def load_data():
    print("Loading embeddings and metadata...")
    # 1. Load Embeddings
    user_embs = np.load(LIGHTGCN_DIR / "user_embeddings.npy")
    book_embs = np.load(LIGHTGCN_DIR / "book_embeddings.npy")
    
    # Normalize book embeddings for pure cosine similarity (Item-Item)
    book_norms = np.linalg.norm(book_embs, axis=1, keepdims=True)
    book_embs_normalized = book_embs / (book_norms + 1e-10)
    
    # 2. Load Mappings & Metadata
    book_map = pd.read_parquet(GRAPH_DIR / "book_mapping.parquet")
    books_clean = pd.read_parquet(PROCESSED_DIR / "books_clean.parquet")
    
    # Merge to create a master lookup dataframe: [book_idx, book_id, title]
    book_lookup = book_map.merge(
        books_clean[['book_id', 'title']], 
        on='book_id', 
        how='left'
    )
    
    # Fill missing titles just in case
    book_lookup['title'] = book_lookup['title'].fillna("Unknown Title")
    
    # Create fast dictionary lookups
    idx_to_title = dict(zip(book_lookup['book_idx'], book_lookup['title']))
    
    print("✓ Data loaded successfully!\n")
    return user_embs, book_embs, book_embs_normalized, book_lookup, idx_to_title

def find_similar_books(query_title, book_embs_normalized, book_lookup, idx_to_title, top_k=10):
    """Finds books with the highest cosine similarity to the query book."""
    # Find the book index by title substring match
    matches = book_lookup[book_lookup['title'].str.contains(query_title, case=False, na=False)]
    
    if matches.empty:
        print(f"Could not find any books matching '{query_title}'")
        return
        
    target_idx = matches.iloc[0]['book_idx']
    target_title = matches.iloc[0]['title']
    
    print(f"--- Books similar to: '{target_title}' ---")
    
    # 1. Get the normalized vector for the target book
    target_vector = book_embs_normalized[target_idx]
    
    # 2. Compute Cosine Similarity against ALL books simultaneously
    # (Since vectors are normalized, dot product == cosine similarity)
    similarities = np.dot(book_embs_normalized, target_vector)
    
    # 3. Get top K indices (ignoring the book itself at index 0)
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    for rank, idx in enumerate(top_indices, 1):
        score = similarities[idx]
        print(f"{rank}. {idx_to_title.get(idx, 'Unknown')} (Score: {score:.3f})")
    print("\n")

def recommend_for_user(user_idx, user_embs, book_embs, idx_to_title, top_k=10):
    """Recommends books for a specific user index based on dot product."""
    if user_idx >= len(user_embs):
        print("User index out of bounds.")
        return
        
    print(f"--- Top {top_k} Recommendations for User #{user_idx} ---")
    
    # 1. Get user vector
    u_vec = user_embs[user_idx]
    
    # 2. Compute LightGCN preference scores (raw dot product)
    scores = np.dot(book_embs, u_vec)
    
    # 3. Get top K indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    for rank, idx in enumerate(top_indices, 1):
        score = scores[idx]
        print(f"{rank}. {idx_to_title.get(idx, 'Unknown')} (Raw Score: {score:.3f})")
    print("\n")

if __name__ == "__main__":
    user_embs, book_embs, book_embs_normalized, book_lookup, idx_to_title = load_data()
    
    # Test 1: Item-Item Similarity
    # Try plugging in famous books from different genres to see if the model groups them correctly
    find_similar_books("Harry Potter and the Sorcerer", book_embs_normalized, book_lookup, idx_to_title)
    find_similar_books("The Hobbit", book_embs_normalized, book_lookup, idx_to_title)
    find_similar_books("Pride and Prejudice", book_embs_normalized, book_lookup, idx_to_title)
    
    # Test 2: User Recommendations
    # Pick a few random user indices to see what the model suggests
    recommend_for_user(user_idx=42, user_embs=user_embs, book_embs=book_embs, idx_to_title=idx_to_title)
    recommend_for_user(user_idx=1024, user_embs=user_embs, book_embs=book_embs, idx_to_title=idx_to_title)