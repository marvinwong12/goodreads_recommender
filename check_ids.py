import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
VECTOR_DB_DIR = PROJECT_ROOT / "models" / "vector_index"

def diagnose_mapping():
    print("--- Loading IDs ---")
    
    # 1. Load Graph DB Mapping
    book_map = pd.read_parquet(GRAPH_DIR / "book_mapping.parquet")
    graph_ids = book_map['book_id'].values
    print(f"Graph DB book_ids (count): {len(graph_ids):,}")
    print(f"Graph DB Data Type: {type(graph_ids[0])}")
    print(f"Graph DB First 5 IDs: {graph_ids[:5]}\n")
    
    # 2. Load Vector DB IDs
    try:
        vector_ids = np.load(VECTOR_DB_DIR / "book_ids.npy", allow_pickle=True)
        print(f"Vector DB book_ids (count): {len(vector_ids):,}")
        print(f"Vector DB Data Type: {type(vector_ids[0])}")
        print(f"Vector DB First 5 IDs: {vector_ids[:5]}\n")
    except Exception as e:
        print(f"Failed to load vector IDs: {e}")
        return

    # 3. Test Match Rates
    print("--- Testing Matches ---")
    
    # Test A: Raw match (what the script is currently doing)
    graph_set_raw = set(graph_ids)
    vector_set_raw = set(vector_ids)
    raw_matches = len(graph_set_raw.intersection(vector_set_raw))
    print(f"Raw Match Count: {raw_matches:,} (If 0, types or formats are misaligned)")
    
    # Test B: Cast both to strings
    graph_set_str = set(str(x) for x in graph_ids)
    vector_set_str = set(str(x) for x in vector_ids)
    str_matches = len(graph_set_str.intersection(vector_set_str))
    print(f"String Match Count: {str_matches:,}")
    
    # Test C: Cast both to ints (ignoring potential non-numeric strings safely)
    def safe_int(val):
        try:
            # Handle floats saved as strings like "1234.0"
            return int(float(val))
        except (ValueError, TypeError):
            return -1

    graph_set_int = set(safe_int(x) for x in graph_ids)
    vector_set_int = set(safe_int(x) for x in vector_ids)
    
    # Remove the error value (-1) from the sets before comparing
    graph_set_int.discard(-1)
    vector_set_int.discard(-1)
    
    int_matches = len(graph_set_int.intersection(vector_set_int))
    print(f"Integer Match Count: {int_matches:,}")

if __name__ == "__main__":
    diagnose_mapping()