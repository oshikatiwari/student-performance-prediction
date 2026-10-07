"""
Model Training and Rigorous Cross-Validation Comparison Module.
Benchmarks 5 distinct regression models under 5-Fold Cross-Validation:
1. Dummy Mean Baseline
2. Ridge Regression (L2 Linear Model)
3. Random Forest Regressor (Bagging Tree Ensemble)
4. Gradient Boosting Regressor (Boosting Tree Ensemble)
5. HistGradientBoosting Regressor (Histogram-based Tree Ensemble)

Reports mean ± std for RMSE, MAE, and R2.
Performs hyperparameter tuning for the top model under a stated budget.
Saves the final fitted pipeline artifact and metrics summary.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

from src.pipeline import build_pipeline, PRE_EXAM_FEATURES


RANDOM_SEED = 42
N_SPLITS = 5


def get_candidate_models() -> Dict[str, Any]:
    """Returns candidate estimators with fixed seeds."""
    return {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_SEED),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=3,
            subsample=0.9,
            random_state=RANDOM_SEED
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=RANDOM_SEED)
    }


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
    """Runs 5-fold cross validation across candidate models reporting mean +/- std."""
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    results = {}

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2"
    }

    print("=" * 80)
    print(f"5-FOLD CROSS VALIDATION MODEL BENCHMARK (Seed = {RANDOM_SEED})")
    print("Strategy: 5-Fold K-Fold | Metric: RMSE, MAE, R-squared (Reporting Mean +/- Std)")
    print("=" * 80)
    print(f"{'Model':<24s} | {'CV RMSE':<18s} | {'CV MAE':<18s} | {'CV R-squared':<18s}")
    print("-" * 80)

    for name, model in get_candidate_models().items():
        pipeline = build_pipeline(model)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        rmse_mean, rmse_std = float(-scores["test_rmse"].mean()), float(scores["test_rmse"].std())
        mae_mean, mae_std = float(-scores["test_mae"].mean()), float(scores["test_mae"].std())
        r2_mean, r2_std = float(scores["test_r2"].mean()), float(scores["test_r2"].std())

        results[name] = {
            "rmse_mean": round(rmse_mean, 3),
            "rmse_std": round(rmse_std, 3),
            "mae_mean": round(mae_mean, 3),
            "mae_std": round(mae_std, 3),
            "r2_mean": round(r2_mean, 3),
            "r2_std": round(r2_std, 3)
        }

        print(f"{name:<24s} | {rmse_mean:.3f} +/- {rmse_std:.3f}   | {mae_mean:.3f} +/- {mae_std:.3f}   | {r2_mean:.3f} +/- {r2_std:.3f}")

    print("=" * 80)
    return results


def tune_best_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """
    Performs hyperparameter tuning for GradientBoosting under a stated budget:
    Budget: 25 iterations x 5-fold CV = 125 model fits.
    """
    print("\n--- Hyperparameter Tuning (Budget: 25 iterations x 5-fold CV) ---")
    param_distributions = {
        "model__n_estimators": [80, 100, 120, 150, 180],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.12],
        "model__max_depth": [2, 3, 4],
        "model__subsample": [0.8, 0.85, 0.9, 0.95, 1.0],
        "model__min_samples_leaf": [2, 4, 6, 8]
    }

    base_pipe = build_pipeline(GradientBoostingRegressor(random_state=RANDOM_SEED))
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        base_pipe,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    search.fit(X, y)

    print(f"Best CV RMSE: {-search.best_score_:.3f}")
    print(f"Best Hyperparameters: {search.best_params_}")
    return search.best_estimator_


def train_and_save(
    data_path: str = "data/student_performance.csv",
    output_dir: str = "artifacts"
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    # Filter invalid target records (bad ground-truth labels)
    clean_mask = (df["FinalExamScore"] >= 0) & (df["FinalExamScore"] <= 100)
    df_clean = df[clean_mask].copy()
    print(f"Loaded {len(df)} rows. Retained {len(df_clean)} clean training records (dropped {len(df) - len(df_clean)} invalid labels).")

    X = df_clean.drop(columns=["FinalExamScore"])
    y = df_clean["FinalExamScore"]

    # 1. Benchmark all candidate approaches
    metrics_summary = evaluate_models(X, y)

    # 2. Hyperparameter tuning on best approach
    best_pipeline = tune_best_model(X, y)

    # 3. Fit on full clean training dataset
    best_pipeline.fit(X, y)

    # 4. Save artifacts
    model_path = os.path.join(output_dir, "model_pipeline.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"\nSaved final fitted pipeline to {model_path}")

    metrics_file = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics summary to {metrics_file}")

    return best_pipeline, metrics_summary


if __name__ == "__main__":
    train_and_save()
