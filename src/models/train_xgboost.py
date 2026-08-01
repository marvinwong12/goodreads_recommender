# src/models/train_xgboost.py
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import GroupShuffleSplit
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
NDCG_K = 10
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'lightgcn_score',
    'semantic_score',
    'fused_score',
    'user_avg_rating',
    'user_explicit_rating_count',
    'author_read_count',
    'book_average_rating',
    'book_log_ratings_count',
    'cooccurrence_score',
]

# Force non-decreasing relevance w.r.t. the stage-1 score features and the
# book popularity priors. Without this, unconstrained trees can fit a
# non-monotonic (and non-extrapolating) function of these scores that only
# holds inside the training window's range, actively scrambling an otherwise-
# good ordering once served on real/future data. `author_read_count` is left
# unconstrained: more prior reads of an author is a plausible but not
# guaranteed-monotonic signal (readers do deliberately branch out).
# 1 = monotonic increasing, 0 = unconstrained.
MONOTONE_FEATURES = {
    'lightgcn_score', 'semantic_score', 'fused_score',
    'book_average_rating', 'book_log_ratings_count', 'cooccurrence_score',
}
MONOTONE_CONSTRAINTS = tuple(1 if f in MONOTONE_FEATURES else 0 for f in FEATURES)


def mean_ndcg_at_k(y_true, y_score, qid, k=NDCG_K):
    """
    Average per-user (per-qid) NDCG@k. Unlike a pooled AUC, this measures
    within-user ranking quality directly, which is what HR@k/NDCG@k in
    production actually depend on.
    """
    frame = pd.DataFrame({"qid": qid, "y_true": y_true, "y_score": y_score})
    ndcgs = []

    for _, group in frame.groupby("qid", sort=False):
        n_pos = int(group["y_true"].sum())
        if n_pos == 0:
            continue

        ranked = group.sort_values("y_score", ascending=False)["y_true"].to_numpy()
        ranks = np.arange(1, len(ranked) + 1)
        dcg = np.sum(ranked[:k] / np.log2(ranks[:k] + 1))

        ideal_hits = min(n_pos, k)
        idcg = np.sum(1.0 / np.log2(np.arange(1, ideal_hits + 1) + 1))

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def objective(trial, X_train, y_train, qid_train, X_val, y_val, qid_val):
    """Optuna objective function to find the best hyperparameters."""

    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'objective': 'rank:ndcg',
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': f'ndcg@{NDCG_K}',
        'early_stopping_rounds': 20,
        'monotone_constraints': MONOTONE_CONSTRAINTS,
        'n_jobs': -1
    }

    model = xgb.XGBRanker(**param_grid)

    model.fit(
        X_train, y_train, qid=qid_train,
        eval_set=[(X_val, y_val)],
        eval_qid=[qid_val],
        verbose=False
    )

    val_preds = model.predict(X_val)
    return mean_ndcg_at_k(y_val, val_preds, qid_val, k=NDCG_K)


def train():
    print("Loading tabular dataset...")
    df = pd.read_parquet(DATA_DIR / "xgboost_dataset.parquet")

    features = FEATURES

    # Ranking objectives require rows for the same query (user) to be
    # contiguous, and train/val must never split a user's candidates across
    # both sides (that would leak the user's other candidates' relative
    # ordering context and also makes qid grouping meaningless).
    df = df.sort_values('user_idx').reset_index(drop=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df['user_idx']))

    df_train = df.iloc[train_idx].sort_values('user_idx').reset_index(drop=True)
    df_val = df.iloc[val_idx].sort_values('user_idx').reset_index(drop=True)

    X_train, y_train, qid_train = df_train[features], df_train['target'], df_train['user_idx']
    X_val, y_val, qid_val = df_val[features], df_val['target'], df_val['user_idx']

    print(f"Training on {len(X_train):,} rows ({qid_train.nunique():,} users). "
          f"Validating on {len(X_val):,} rows ({qid_val.nunique():,} users)...")
    print(f"\n--- Starting Optuna Hyperparameter Tuning ({NUM_OPTUNA_TRIALS} Trials) ---")

    # Create the Optuna study (we want to maximize mean NDCG@10)
    study = optuna.create_study(direction='maximize', study_name="xgboost_ranker")

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, qid_train, X_val, y_val, qid_val),
        n_trials=NUM_OPTUNA_TRIALS
    )

    print("\n====================================")
    print(f"✓ Tuning Complete! Best Mean NDCG@{NDCG_K}: {study.best_value:.4f}")
    print("====================================\n")

    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # --- Train Final Model ---
    print("\nTraining final model with best parameters...")
    final_params = study.best_params
    final_params['objective'] = 'rank:ndcg'
    final_params['tree_method'] = 'hist'
    final_params['device'] = 'cuda'
    final_params['eval_metric'] = f'ndcg@{NDCG_K}'
    final_params['early_stopping_rounds'] = 20
    final_params['monotone_constraints'] = MONOTONE_CONSTRAINTS

    best_model = xgb.XGBRanker(**final_params)
    best_model.fit(
        X_train, y_train, qid=qid_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_qid=[qid_train, qid_val],
        verbose=100  # Show progress every 100 trees
    )

    # --- Save Artifacts ---
    model_path = MODEL_DIR / "ranker_model.json"
    best_model.save_model(model_path)
    print(f"\n✓ Best model saved to {model_path}")

    final_val_preds = best_model.predict(X_val)
    final_ndcg = mean_ndcg_at_k(y_val, final_val_preds, qid_val, k=NDCG_K)
    print(f"✓ Final validation Mean NDCG@{NDCG_K}: {final_ndcg:.4f}")

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
