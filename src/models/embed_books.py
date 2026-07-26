# src/models/embed_books.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTOR_DIR = PROCESSED_DIR / "vector_index"

# Lightweight, highly effective sentence transformer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256

def build_vector_index():
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load cleaned books metadata
    books_path = PROCESSED_DIR / "books_clean.parquet"
    print(f"Loading books metadata from {books_path}...")
    df_books = pd.read_parquet(books_path, columns=["book_id", "title", "text_for_embedding"])
    
    # Fallback for any unexpected nulls
    df_books["text_for_embedding"] = df_books["text_for_embedding"].fillna(df_books["title"]).fillna("")
    
    book_ids = df_books["book_id"].values
    texts = df_books["text_for_embedding"].tolist()
    num_books = len(df_books)
    
    print(f"Loaded {num_books:,} books for embedding generation.")
    
    # 2. Load Sentence Transformer model
    print(f"\nLoading Transformer Model: '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 3. Generate Embeddings
    print("Generating dense embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # Crucial for Cosine Similarity via Inner Product
        convert_to_numpy=True
    )
    
    embedding_dim = embeddings.shape[1]
    print(f"✓ Generated embedding matrix shape: {embeddings.shape} (Dim: {embedding_dim})")
    
    # 4. Build FAISS Vector Index
    print("\nBuilding FAISS IndexFlatIP (Inner Product / Cosine Similarity)...")
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    
    print(f"✓ Added {index.ntotal:,} vectors to FAISS index.")
    
    # 5. Save Artifacts to Disk
    faiss_file = VECTOR_DIR / "books_faiss.index"
    book_ids_file = VECTOR_DIR / "book_ids.npy"
    embeddings_file = VECTOR_DIR / "book_embeddings.npy"
    meta_file = VECTOR_DIR / "vector_meta.json"
    
    print("\nSaving vector index and artifacts...")
    faiss.write_index(index, str(faiss_file))
    np.save(book_ids_file, book_ids)
    np.save(embeddings_file, embeddings)
    
    meta = {
        "model_name": MODEL_NAME,
        "num_books": num_books,
        "embedding_dim": embedding_dim,
        "faiss_file": faiss_file.name,
        "book_ids_file": book_ids_file.name
    }
    
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=4)
        
    print("==========================================")
    print("✓ Semantic Vector Search Index Built!")
    print(f"  FAISS Index File : {faiss_file}")
    print(f"  Book IDs Mapping : {book_ids_file}")
    print(f"  Embeddings Matrix: {embeddings_file}")
    print("==========================================")

if __name__ == "__main__":
    build_vector_index()