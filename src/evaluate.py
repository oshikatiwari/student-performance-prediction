"""
Error Analysis and Residual Diagnostics Module.
Generates out-of-fold predictions using 5-Fold Cross-Validation:
- Computes residuals: y_true - y_pred
- Produces diagnostic plots: Residuals vs Predicted, Actual vs Predicted, Residual Distribution
- Performs subgroup/segment error analysis across student cohorts
- Identifies and diagnoses failing segments (e.g. high-backlog students and extreme score tails)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_predict
from src.pipeline import PRE_EXAM_FEATURES


def perform_error_analysis(
    data_path: str = "data/student_performance.csv",
    model_path: str = "artifacts/model_pipeline.joblib",
    output_dir: str = "artifacts"
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    # Clean valid target records
    df_clean = df[(df["FinalExamScore"] >= 0) & (df["FinalExamScore"] <= 100)].copy()
    X = df_clean.drop(columns=["FinalExamScore"])
    y = df_clean["FinalExamScore"]

    pipeline = joblib.load(model_path)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Out-of-fold predictions ensure unbiased residual estimates
    y_pred_oof = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
    y_pred_oof = np.clip(y_pred_oof, 0.0, 100.0)

    residuals = y.values - y_pred_oof
    abs_errors = np.abs(residuals)

    df_eval = df_clean.copy()
    df_eval["y_pred"] = y_pred_oof
    df_eval["residual"] = residuals
    df_eval["abs_error"] = abs_errors

    # 1. Generate Diagnostic Residual Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_theme(style="whitegrid")

    # Subplot A: Actual vs Predicted
    axes[0].scatter(y, y_pred_oof, alpha=0.55, color="#1f77b4", edgecolors="none")
    axes[0].plot([20, 100], [20, 100], color="#d62728", linestyle="--", linewidth=2, label="Ideal (y = y_hat)")
    axes[0].set_title("Actual vs Predicted Score (Out-of-Fold)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Actual FinalExamScore")
    axes[0].set_ylabel("Predicted FinalExamScore")
    axes[0].legend()

    # Subplot B: Residuals vs Predicted
    axes[1].scatter(y_pred_oof, residuals, alpha=0.55, color="#2ca02c", edgecolors="none")
    axes[1].axhline(0, color="#d62728", linestyle="--", linewidth=2)
    axes[1].set_title("Residual Plot (Residuals vs Predicted)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted FinalExamScore")
    axes[1].set_ylabel("Residual (Actual - Predicted)")

    # Subplot C: Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[2], color="#9467bd", bins=25)
    axes[2].axvline(0, color="#d62728", linestyle="--", linewidth=2)
    axes[2].set_title(f"Residual Distribution (Mean: {np.mean(residuals):.2f}, Std: {np.std(residuals):.2f})", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Residual")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "residual_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved residual diagnostics plot to {plot_path}")

    # 2. Segment Analysis
    print("\n" + "=" * 80)
    print("                      SUBGROUP / SEGMENT ERROR ANALYSIS")
    print("=" * 80)

    # Segment A: Performance by Previous Backlogs
    print("\n--- Segment A: Error by Previous Backlogs Count ---")
    backlog_summary = df_eval.groupby(pd.cut(df_eval["PreviousBacklogs"], bins=[-1, 0, 1, 2, 10], labels=["0 Backlogs", "1 Backlog", "2 Backlogs", "3+ Backlogs"]))[["abs_error", "residual"]].agg(["count", "mean", "std"])
    print(backlog_summary)

    # Segment B: Performance by Score Tier
    print("\n--- Segment B: Error by Target Exam Score Tier ---")
    score_bins = pd.cut(df_eval["FinalExamScore"], bins=[0, 60, 75, 90, 100], labels=["Low (<60)", "Average (60-75)", "Good (75-90)", "Top Tier (>90)"])
    tier_summary = df_eval.groupby(score_bins)[["abs_error", "residual"]].agg(["count", "mean", "std"])
    print(tier_summary)

    # Segment C: Attendance Tiers
    print("\n--- Segment C: Error by Attendance Tier ---")
    att_bins = pd.cut(df_eval["AttendancePercentage"], bins=[0, 65, 80, 105], labels=["Low (<65%)", "Moderate (65-80%)", "High (>80%)"])
    att_summary = df_eval.groupby(att_bins)[["abs_error", "residual"]].agg(["count", "mean", "std"])
    print(att_summary)

    print("=" * 80)
    return df_eval


if __name__ == "__main__":
    perform_error_analysis()
