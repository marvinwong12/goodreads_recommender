# src/data/make_graph_data.py
from pathlib import Path
import json
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def build_k_core_graph(k: int = 10):
    spark = (
        SparkSession.builder
        .appName("MakeKCoreGraph")
        .config("spark.driver.memory", "10g")
        .getOrCreate()
    )
    
    print("Loading clean interactions...")
    df_interactions = spark.read.parquet(str(PROCESSED_DIR / "interactions_clean.parquet"))
    
    # Rule 1: GCN-fit-only edges with positive engagement (read or rating >= 3).
    # Deliberately uses `is_gcn_fit` (a strict subset of `is_train`) rather than
    # `is_train` so the interactions held back for XGBoost ranker labels
    # (is_train & ~is_gcn_fit) are never seen by LightGCN during training.
    # Otherwise the BPR loss directly optimizes pos_score - neg_score on the
    # exact edges that later become the ranker's positive labels, and the
    # resulting lightgcn_score becomes a near-perfect (leaked) predictor.
    train_edges = df_interactions.filter(
        (F.col("is_gcn_fit") == True) & (F.col("is_engagement") == 1)
    ).select("user_id", "book_id")
    
    initial_count = train_edges.count()
    print(f"Initial Training Edges: {initial_count:,}")
    
    # Rule 2: Iterative K-Core Filtering
    print(f"\nApplying Iterative {k}-Core Filtering...")
    prev_count = 0
    curr_count = initial_count
    iteration = 1
    
    while curr_count != prev_count:
        prev_count = curr_count
        
        # Keep users with >= K interactions
        valid_users = (
            train_edges.groupBy("user_id")
            .count()
            .filter(F.col("count") >= k)
            .select("user_id")
        )
        train_edges = train_edges.join(valid_users, on="user_id", how="inner")
        
        # Keep books with >= K interactions
        valid_books = (
            train_edges.groupBy("book_id")
            .count()
            .filter(F.col("count") >= k)
            .select("book_id")
        )
        train_edges = train_edges.join(valid_books, on="book_id", how="inner")
        
        curr_count = train_edges.count()
        print(f"  Iteration {iteration} | Edges remaining: {curr_count:,}")
        iteration += 1

    # Rule 3: Contiguous Integer Mapping for PyTorch Geometric (Distributed zipWithIndex)
    print("\nMapping User & Book IDs to contiguous integers [0..N-1]...")
    
    # Unique Users Map (Distributed without single-partition bottlenecks)
    unique_users_rdd = train_edges.select("user_id").distinct().rdd.map(lambda r: r.user_id)
    users_df = unique_users_rdd.zipWithIndex().toDF(["user_id", "user_idx"])
    
    # Unique Books Map (Distributed)
    unique_books_rdd = train_edges.select("book_id").distinct().rdd.map(lambda r: r.book_id)
    books_df = unique_books_rdd.zipWithIndex().toDF(["book_id", "book_idx"])
    
    num_users = users_df.count()
    num_books = books_df.count()
    
    # Join integer indices back to edges
    mapped_edges = (
        train_edges
        .join(users_df, on="user_id", how="inner")
        .join(books_df, on="book_id", how="inner")
        .select("user_idx", "book_idx", "user_id", "book_id")
    )
    
    # Save graph edge index parquet
    graph_out_dir = PROCESSED_DIR / "graph"
    graph_out_dir.mkdir(parents=True, exist_ok=True)
    
    mapped_edges.write.mode("overwrite").parquet(str(graph_out_dir / f"edges_k{k}.parquet"))
    
    # Save ID Mappings for inference lookup
    users_df.write.mode("overwrite").parquet(str(graph_out_dir / "user_mapping.parquet"))
    books_df.write.mode("overwrite").parquet(str(graph_out_dir / "book_mapping.parquet"))
    
    # Save metadata summary JSON
    meta = {
        "k_core": k,
        "num_users": num_users,
        "num_books": num_books,
        "num_edges": curr_count,
        "sparsity_percent": float(100 * (1 - curr_count / (num_users * num_books)))
    }
    
    with open(graph_out_dir / "graph_meta.json", "w") as f:
        json.dump(meta, f, indent=4)
        
    print("\n==========================================")
    print(f"✓ K-Core Graph Successfully Built!")
    print(f"  Users Node Count : {num_users:,}")
    print(f"  Books Node Count : {num_books:,}")
    print(f"  Total Graph Edges: {curr_count:,}")
    print(f"  Graph Sparsity   : {meta['sparsity_percent']:.4f}%")
    print(f"  Saved to         : {graph_out_dir}")
    print("==========================================")
    
    spark.stop()

if __name__ == "__main__":
    build_k_core_graph(k=10)