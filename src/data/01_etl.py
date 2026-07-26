# src/data/etl.py
import argparse
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType, FloatType, StringType, BooleanType, StructType, StructField
)

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

DEFAULT_RAW_DIR = str(PROJECT_ROOT / "data" / "raw")
DEFAULT_OUT_DIR = str(PROJECT_ROOT / "data" / "interim")

def get_spark_session(app_name: str = "GoodreadsETL") -> SparkSession:
    """Initializes local PySpark session with optimized memory settings."""
    return (
        SparkSession.builder.appName(app_name)
        # Bump memory slightly and allow unlimited result size
        .config("spark.driver.memory", "10g")
        .config("spark.driver.maxResultSize", "0") 
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )

def process_data(raw_dir: str = DEFAULT_RAW_DIR, output_dir: str = DEFAULT_OUT_DIR):
    spark = get_spark_session()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # 1. Process Interactions
    # ==========================================
    print("Processing Interactions...")
    
    # Explicit schema prevents Spark from crashing your RAM during inference
    interactions_schema = StructType([
        StructField("user_id", StringType(), True),
        StructField("book_id", StringType(), True),
        StructField("is_read", BooleanType(), True),
        StructField("rating", IntegerType(), True),
        StructField("date_updated", StringType(), True),
    ])
    
    # Read using the explicit schema
    interactions_df = spark.read.schema(interactions_schema).json(
        f"{raw_dir}/goodreads_interactions_fantasy_paranormal.json.gz"
    )
    
    cleaned_interactions = interactions_df.select(
        F.col("user_id"),
        F.col("book_id"),
        F.col("rating"),
        F.col("is_read").cast(IntegerType()),
        F.to_timestamp(
            F.concat_ws(
                " ", 
                F.substring("date_updated", 1, 19), 
                F.substring("date_updated", 27, 4)
            ), 
            "EEE MMM dd HH:mm:ss yyyy"
        ).alias("interaction_date")
    )
    cleaned_interactions.write.mode("overwrite").parquet(f"{output_dir}/interactions.parquet")

    # ==========================================
    # 2. Process Books Metadata
    # ==========================================
    print("Processing Books Metadata...")
    books_df = spark.read.json(f"{raw_dir}/goodreads_books_fantasy_paranormal.json.gz")
    
    # Helper to convert empty strings to None (Null) before casting
    def clean_cast(c, t):
        return F.when(F.col(c) == "", None).otherwise(F.col(c)).cast(t)

    cleaned_books = books_df.select(
        F.col("book_id"),
        F.col("title"),
        F.col("description"),
        clean_cast("average_rating", FloatType()).alias("average_rating"),
        clean_cast("ratings_count", IntegerType()).alias("ratings_count"),
        clean_cast("publication_year", IntegerType()).alias("publication_year"),
        clean_cast("num_pages", IntegerType()).alias("num_pages"),
        F.col("image_url"),
        F.col("authors.author_id").alias("author_ids")
    )
    cleaned_books.write.mode("overwrite").parquet(f"{output_dir}/books.parquet")

    # ==========================================
    # 3. Process Reviews
    # ==========================================
    print("Processing Reviews...")
    reviews_df = spark.read.json(f"{raw_dir}/goodreads_reviews_fantasy_paranormal.json.gz")
    
    cleaned_reviews = reviews_df.select(
        F.col("user_id"),
        F.col("book_id"),
        F.col("review_id"),
        F.col("rating").cast(IntegerType()),
        F.col("n_votes").cast(IntegerType()),
        F.col("review_text")
    )
    cleaned_reviews.write.mode("overwrite").parquet(f"{output_dir}/reviews.parquet")

    print(f"ETL Complete! Cleaned Parquet files saved to {output_dir}")
    spark.stop()

if __name__ == "__main__":
    process_data()