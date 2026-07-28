# src/data/clean_data.py
from pathlib import Path
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

INTERIM_DIR = str(PROJECT_ROOT / "data" / "interim")
PROCESSED_DIR = str(PROJECT_ROOT / "data" / "processed")

# Temporal split cutoff date based on EDA timeline audit
TEMPORAL_CUTOFF_DATE = "2017-07-01 00:00:00"

def clean_and_process():
    spark = (
        SparkSession.builder
        .appName("GoodreadsDataCleaning")
        .config("spark.driver.memory", "10g")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )
    
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    
    # Clean Books Metadata
    print("Cleaning Books Metadata...")
    df_books = spark.read.parquet(f"{INTERIM_DIR}/books.parquet")
    
    cleaned_books = (
        df_books
        # Fix 1: Nullify unrealistic publication years (< 1800 or > 2018)
        .withColumn(
            "publication_year",
            F.when(
                (F.col("publication_year") >= 1800) & (F.col("publication_year") <= 2018),
                F.col("publication_year")
            ).otherwise(None)
        )
        # Fix 2: Create fallback text for embeddings (Title if description < 20 chars or null)
        .withColumn(
            "text_for_embedding",
            F.when(
                (F.col("description").isNotNull()) & (F.length(F.trim(F.col("description"))) >= 20),
                F.col("description")
            ).otherwise(F.col("title"))
        )
        # Feature 1: Page count binary flag
        .withColumn("is_long_book", F.when(F.col("num_pages") > 400, 1).otherwise(0))
        # Feature 2: Safe primary author extraction guarded by size check
        .withColumn(
            "primary_author_id",
            F.when(
                F.size(F.col("author_ids")) > 0, 
                F.col("author_ids").getItem(0)
            ).otherwise(None)
        )
    )
    
    cleaned_books.write.mode("overwrite").parquet(f"{PROCESSED_DIR}/books_clean.parquet")
    print(f"✓ Saved cleaned books to {PROCESSED_DIR}/books_clean.parquet")

    # Compute User Statistics & Clean Interactions
    print("Processing User Rating Bias & Interaction Features...")
    df_interactions = spark.read.parquet(f"{INTERIM_DIR}/interactions.parquet")
    
    # 1. Extract temporal split FIRST
    interactions_cleaned = (
        df_interactions
        .withColumn("interaction_year", F.year("interaction_date"))
        .withColumn("is_train", F.col("interaction_date") < F.lit(TEMPORAL_CUTOFF_DATE))
    )
    
    # 2. Calculate user stats ONLY on the training data to prevent leakage
    explicit_train_ratings = interactions_cleaned.filter(
        (F.col("rating") > 0) & (F.col("is_train") == True)
    )
    
    user_stats = explicit_train_ratings.groupBy("user_id").agg(
        F.mean("rating").cast(FloatType()).alias("user_avg_rating"),
        F.count("rating").alias("user_explicit_rating_count")
    )
    
    # Global average rating for cold-start users
    global_avg_row = explicit_train_ratings.select(F.mean("rating")).first()
    global_avg_rating = float(global_avg_row[0]) if global_avg_row else 3.95
    
    # 3. Join user stats back into the FULL interactions dataset
    interactions_cleaned = interactions_cleaned.join(user_stats, on="user_id", how="left")
    
    interactions_cleaned = interactions_cleaned.fillna({
        "user_avg_rating": global_avg_rating,
        "user_explicit_rating_count": 0
    })
    
    # 4. Join publication year
    book_years = cleaned_books.select("book_id", "publication_year")
    interactions_cleaned = interactions_cleaned.join(book_years, on="book_id", how="left")
    
    # 5. Apply remaining feature engineering
    interactions_final = (
        interactions_cleaned
        .withColumn(
            "rating_delta",
            F.when(F.col("rating") > 0, F.col("rating") - F.col("user_avg_rating")).otherwise(0.0)
        )
        .withColumn(
            "is_engagement",
            F.when((F.col("is_read") == 1) | (F.col("rating") >= 3), 1).otherwise(0)
        )
        .withColumn(
            "book_age_at_interaction",
            F.when(
                F.col("publication_year").isNotNull(),
                F.greatest(F.lit(0), F.col("interaction_year") - F.col("publication_year"))
            ).otherwise(None)
        )
        .drop("publication_year")
    )
    
    # Drop temp join column to keep dataset lean
    interactions_final = interactions_cleaned.drop("publication_year")
    
    interactions_final.write.mode("overwrite").parquet(f"{PROCESSED_DIR}/interactions_clean.parquet")
    print(f"✓ Saved cleaned interactions to {PROCESSED_DIR}/interactions_clean.parquet")
    
    spark.stop()
    print("\nData preprocessing & feature engineering complete!")

if __name__ == "__main__":
    clean_and_process()