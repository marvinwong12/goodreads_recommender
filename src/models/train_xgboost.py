# src/models/train_xgboost.py
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "xgboost"

def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading tabular dataset...")
    df = pd.read_parquet(DATA_DIR / "xgboost_dataset.parquet")
    
    # Define our features and target
    features = [
        'lightgcn_score', 
        'semantic_score', 
        'user_avg_rating', 
        'user_explicit_rating_count',
        'is_long_book',
        'book_age_at_interaction'
    ]
    
    X = df[features]
    y = df['target']
    
    # Stratified split ensures the 1-to-0 ratio remains identical in both sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training on {len(X_train):,} rows. Validating on {len(X_val):,} rows...")
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        objective='binary:logistic',
        tree_method='hist', # Change to 'gpu_hist' if running on Colab GPU
        eval_metric='auc',
        early_stopping_rounds=20
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )
    
    # Evaluate
    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    print(f"\n====================================")
    print(f"✓ Validation AUC Score: {auc:.4f}")
    print(f"====================================\n")
    
    # Save the model
    model_path = MODEL_DIR / "ranker_model.json"
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot feature importance
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "feature_importance.png")
    print("Feature importance plot saved.")

if __name__ == "__main__":
    train()