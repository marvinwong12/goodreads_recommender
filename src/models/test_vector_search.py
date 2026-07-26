# src/models/test_vector_search.py
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTOR_DIR = PROCESSED_DIR / "vector_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_vector_search_engine():
    print("Loading FAISS index, book mapping, and Transformer model...")
    
    # Load index and ID mappings
    index = faiss.read_index(str(VECTOR_DIR / "books_faiss.index"))
    book_ids = np.load(VECTOR_DIR / "book_ids.npy", allow_pickle=True)
    
    # Load metadata for display
    df_books = pd.read_parquet(
        PROCESSED_DIR / "books_clean.parquet", 
        columns=["book_id", "title", "description"]
    )
    book_meta = df_books.set_index("book_id").to_dict(orient="index")
    
    # Load encoder model
    model = SentenceTransformer(MODEL_NAME)
    
    print("✓ Vector engine loaded successfully!\n")
    return index, book_ids, book_meta, model

def search_by_prompt(query_text, index, book_ids, book_meta, model, top_k=5):
    print("=" * 80)
    print(f"QUERY: '{query_text}'")
    print("=" * 80)
    
    # Embed query vector and normalize for cosine similarity
    query_vector = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
    
    # Retrieve top K closest vectors from FAISS
    scores, indices = index.search(query_vector, top_k)
    
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        b_id = book_ids[idx]
        meta = book_meta.get(b_id, {})
        title = meta.get("title", "Unknown Title")
        desc = meta.get("description", "")
        desc_snippet = (desc[:140] + "...") if desc and len(desc) > 140 else (desc or "No description available.")
        
        print(f"Rank {rank} | Cosine Similarity Score: {score:.4f}")
        print(f"  Title   : {title}")
        print(f"  Book ID : {b_id}")
        print(f"  Snippet : {desc_snippet}")
        print("-" * 80)

def main():
    index, book_ids, book_meta, model = load_vector_search_engine()
    
    # Test Prompt 1: Dark Fantasy / Magic System
    prompt_1 = "A dark fantasy novel about assassin mages, secret guilds, and ancient blood magic"
    search_by_prompt(prompt_1, index, book_ids, book_meta, model, top_k=5)
    
    # Test Prompt 2: Epic Dragon High Fantasy
    prompt_2 = "An epic high fantasy adventure featuring dragon riders, kingdom wars, and political intrigue"
    search_by_prompt(prompt_2, index, book_ids, book_meta, model, top_k=5)

if __name__ == "__main__":
    main()