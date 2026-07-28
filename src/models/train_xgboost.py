# src/models/train_xgboost.py
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings

# Suppress XGBoost future warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning)

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "xgboost"

# --- Configuration ---
NUM_OPTUNA_TRIALS = 30  # Increase to 50-100 for overnight training
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight):
    """Optuna objective function to find the best hyperparameters."""
    
    # 1. Define the hyperparameter search space
    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': scale_pos_weight,
        'objective': 'binary:logistic',
        'tree_method': 'hist', # Change to 'gpu_hist' if using Colab/GPU
        'eval_metric': 'auc',
        'early_stopping_rounds': 20,
        'n_jobs': -1 # Use all CPU cores
    }
    
    # 2. Initialize and train the model
    model = xgb.XGBClassifier(**param_grid)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False # Keep terminal clean during tuning
    )
    
    # 3. Evaluate and return AUC
    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    
    return auc

def train():
    print("Loading tabular dataset...")
    df = pd.read_parquet(DATA_DIR / "xgboost_dataset.parquet")
    
    # Added 'fused_score' to the features!
    features = [
        'lightgcn_score', 
        'semantic_score', 
        'fused_score', 
        'user_avg_rating', 
        'user_explicit_rating_count',
        'is_long_book',
        'book_age_at_interaction'
    ]
    
    X = df[features]
    y = df['target']
    
    # Calculate class imbalance weight
    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()
    scale_pos_weight = num_neg / num_pos
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f} (Positives: {num_pos:,}, Negatives: {num_neg:,})")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training on {len(X_train):,} rows. Validating on {len(X_val):,} rows...")
    print(f"\n--- Starting Optuna Hyperparameter Tuning ({NUM_OPTUNA_TRIALS} Trials) ---")
    
    # Create the Optuna study (we want to maximize AUC)
    study = optuna.create_study(direction='maximize', study_name="xgboost_ranker")
    
    # Run the optimization
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight),
        n_trials=NUM_OPTUNA_TRIALS
    )
    
    print("\n====================================")
    print(f"✓ Tuning Complete! Best AUC: {study.best_value:.4f}")
    print("====================================\n")
    
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # --- Train Final Model ---
    print("\nTraining final model with best parameters...")
    final_params = study.best_params
    final_params['scale_pos_weight'] = scale_pos_weight
    final_params['objective'] = 'binary:logistic'
    final_params['tree_method'] = 'hist'
    final_params['eval_metric'] = 'auc'
    final_params['early_stopping_rounds'] = 20
    
    best_model = xgb.XGBClassifier(**final_params)
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100 # Show progress every 100 trees
    )
    
    # --- Save Artifacts ---
    model_path = MODEL_DIR / "ranker_model.json"
    best_model.save_model(model_path)
    print(f"\n✓ Best model saved to {model_path}")
    
    # Plot feature importance
    importance = best_model.feature_importances_
    sorted_idx = np.argsort(importance)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.title('XGBoost Feature Importance (Tuned Model)')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "feature_importance.png")
    print("✓ Feature importance plot saved.")

if __name__ == "__main__":
    train()