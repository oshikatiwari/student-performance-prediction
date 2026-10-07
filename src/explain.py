"""
Feature Interpretability and Permutation Importance Module (Bonus Deliverable).
Computes permutation feature importance on validation data to quantitatively rank
the influence of justified pre-exam features on predicted student performance.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance


def compute_feature_importance(
    data_path: str = "data/student_performance.csv",
    model_path: str = "artifacts/model_pipeline.joblib",
    output_dir: str = "artifacts"
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    df_clean = df[(df["FinalExamScore"] >= 0) & (df["FinalExamScore"] <= 100)].copy()

    X = df_clean.drop(columns=["FinalExamScore"])
    y = df_clean["FinalExamScore"]

    pipeline = joblib.load(model_path)

    # Compute permutation importance
    r = permutation_importance(
        pipeline, X, y,
        n_repeats=15,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    feature_names = [
        "StudyHours", "AttendancePercentage", "PreviousExamScore",
        "AssignmentsCompleted", "SleepHours", "ExtracurricularHours",
        "ClassParticipation", "PreviousBacklogs"
    ]

    # Map importances back to justified features
    # Note: X has full dataframe columns, but pipeline preprocessor only reads pre-exam features
    feature_importances = []
    for col in feature_names:
        idx = list(X.columns).index(col)
        feature_importances.append({
            "feature": col,
            "importance_mean": float(r.importances_mean[idx]),
            "importance_std": float(r.importances_std[idx])
        })

    df_imp = pd.DataFrame(feature_importances).sort_values("importance_mean", ascending=True)

    print("=" * 60)
    print("      PERMUTATION FEATURE IMPORTANCE (Decrease in RMSE)")
    print("=" * 60)
    for _, row in df_imp.sort_values("importance_mean", ascending=False).iterrows():
        print(f"  {row['feature']:<22s}: {row['importance_mean']:6.3f} +/- {row['importance_std']:5.3f}")
    print("=" * 60)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    plt.barh(df_imp["feature"], df_imp["importance_mean"], xerr=df_imp["importance_std"], color="#3b82f6", capsize=4)
    plt.title("Permutation Feature Importance (Metric: Negative RMSE Degradation)", fontsize=13, fontweight="bold")
    plt.xlabel("Permutation Importance Mean")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to {plot_path}")

    return df_imp


if __name__ == "__main__":
    compute_feature_importance()
