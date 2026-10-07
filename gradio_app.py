"""
Gradio Web Application for Student Performance Prediction.
GDG NMIT - Machine Learning Technical Round 2
"""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import gradio as gr

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "model_pipeline.joblib"
RESIDUAL_PLOT_PATH = ROOT / "artifacts" / "residual_plot.png"
IMPORTANCE_PLOT_PATH = ROOT / "artifacts" / "feature_importance.png"

RMSE_ESTIMATE = 6.675
CLASS_MEDIAN = 73.80

# Load trained pipeline
if MODEL_PATH.exists():
    pipeline = joblib.load(MODEL_PATH)
else:
    pipeline = None


def predict_single(
    previous_score: float,
    attendance: float,
    assignments: float,
    backlogs: float,
    study_hours: float,
    sleep_hours: float,
    participation: float,
    extracurricular: float
):
    if pipeline is None:
        return 0.0, "Model not loaded", "[0.0, 0.0]", "Error: Model artifact not found."

    input_df = pd.DataFrame([{
        "StudyHours": study_hours,
        "AttendancePercentage": attendance,
        "PreviousExamScore": previous_score,
        "AssignmentsCompleted": assignments,
        "SleepHours": sleep_hours,
        "ExtracurricularHours": extracurricular,
        "ClassParticipation": participation,
        "PreviousBacklogs": backlogs
    }])

    raw_pred = pipeline.predict(input_df)[0]
    score = float(np.clip(raw_pred, 0.0, 100.0))

    ci_lower = max(0.0, round(score - 1.96 * RMSE_ESTIMATE, 1))
    ci_upper = min(100.0, round(score + 1.96 * RMSE_ESTIMATE, 1))
    interval_str = f"[{ci_lower} , {ci_upper}]"

    delta = score - CLASS_MEDIAN
    delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"

    if score >= 75.0:
        tier = "🌟 High Distinction"
        summary = (
            f"### ✅ **High Academic Standing**\n\n"
            f"- **Predicted Score:** `{score:.2f} / 100` ({delta_str} vs class median)\n"
            f"- **95% Confidence Bounds:** `{interval_str}`\n"
            f"- **Recommendation:** Excellent readiness. Maintain current coursework study discipline."
        )
    elif score >= 50.0:
        tier = "✅ Passing / Satisfactory"
        summary = (
            f"### ℹ️ **Satisfactory Academic Standing**\n\n"
            f"- **Predicted Score:** `{score:.2f} / 100` ({delta_str} vs class median)\n"
            f"- **95% Confidence Bounds:** `{interval_str}`\n"
            f"- **Recommendation:** On track to pass. Targeting study hours towards past exam weak spots can elevate to distinction."
        )
    else:
        tier = "⚠️ At-Risk of Failure"
        summary = (
            f"### 🚨 **At-Risk Alert**\n\n"
            f"- **Predicted Score:** `{score:.2f} / 100` ({delta_str} vs class median)\n"
            f"- **95% Confidence Bounds:** `{interval_str}`\n"
            f"- **Recommendation:** Urgent academic advising recommended. Prioritize clearing pending backlogs and attending review sessions."
        )

    return round(score, 2), tier, interval_str, summary


def predict_batch(file_obj):
    if file_obj is None or pipeline is None:
        return None, None

    # Handle file path
    file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
    df_in = pd.read_csv(file_path)

    if "ID" in df_in.columns:
        ids = df_in["ID"]
    else:
        ids = pd.Series(range(100001, 100001 + len(df_in)), name="ID")

    preds = np.clip(pipeline.predict(df_in), 0.0, 100.0)
    out_df = pd.DataFrame({
        "ID": ids,
        "FinalExamScore": np.round(preds, 2)
    })

    out_csv = ROOT / "submission.csv"
    out_df.to_csv(out_csv, index=False)

    return out_df.head(10), str(out_csv)


# Build clean Gradio interface
with gr.Blocks(title="Student Performance Predictor") as demo:
    gr.Markdown("# 🎓 Student Performance Predictor")
    gr.Markdown("Forecast final exam scores, confidence intervals, and student risk tiers using verified pre-exam indicators.")

    with gr.Tabs():
        # Tab 1: Single Prediction
        with gr.TabItem("⚡ Single Student Prediction"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📚 Academic Track Record")
                    prev_score_input = gr.Slider(0.0, 100.0, value=72.0, step=0.5, label="Previous Exam Score (0–100)")
                    attendance_input = gr.Slider(0.0, 100.0, value=82.0, step=1.0, label="Attendance Percentage (%)")
                    assign_input = gr.Slider(0.0, 100.0, value=85.0, step=1.0, label="Assignments Completed (%)")
                    backlogs_input = gr.Number(value=0, precision=0, label="Previous Pending Backlogs")

                with gr.Column():
                    gr.Markdown("### 🕒 Preparation & Engagement")
                    study_input = gr.Slider(0.0, 16.0, value=5.0, step=0.5, label="Daily Study Hours")
                    sleep_input = gr.Slider(3.0, 12.0, value=7.0, step=0.5, label="Daily Sleep Duration (Hours)")
                    particip_input = gr.Slider(0.0, 10.0, value=6.5, step=0.5, label="Class Participation Score (0–10)")
                    extra_input = gr.Slider(0.0, 20.0, value=4.0, step=0.5, label="Weekly Extracurricular Hours")

            predict_button = gr.Button("🚀 Predict Final Exam Score", variant="primary")

            with gr.Row():
                score_out = gr.Number(label="Predicted Score (/ 100)")
                tier_out = gr.Textbox(label="Academic Standing")
                interval_out = gr.Textbox(label="95% Prediction Interval")

            summary_out = gr.Markdown()

            predict_button.click(
                fn=predict_single,
                inputs=[
                    prev_score_input, attendance_input, assign_input, backlogs_input,
                    study_input, sleep_input, particip_input, extra_input
                ],
                outputs=[score_out, tier_out, interval_out, summary_out]
            )

        # Tab 2: Batch CSV Scoring
        with gr.TabItem("📁 Batch CSV Scoring"):
            gr.Markdown("### Upload Student CSV to Generate `submission.csv`")
            file_upload = gr.File(label="Upload CSV", file_types=[".csv"])
            batch_button = gr.Button("⚡ Generate Predictions", variant="primary")

            preview_table = gr.Dataframe(label="Prediction Preview (First 10 Rows)")
            download_file = gr.File(label="Download Generated Submission CSV")

            batch_button.click(
                fn=predict_batch,
                inputs=[file_upload],
                outputs=[preview_table, download_file]
            )

        # Tab 3: Model Diagnostics & Model Card
        with gr.TabItem("📊 Model Diagnostics & Card"):
            gr.Markdown("""
            ### 📌 Architecture Summary
            - **Pipeline**: Scikit-Learn `Pipeline` with `SimpleImputer` (median), `StandardScaler`, and Tuned `GradientBoostingRegressor`.
            - **5-Fold Cross-Validation**: RMSE **6.675 ± 0.494** | MAE **5.276 ± 0.376** | R² **0.764 ± 0.021**.
            - **Data Leakage Check**: `PostExamConfidence` strictly removed because it occurs *after* the exam.
            - **Intended Use**: Proactive academic advisory support; not for automated punitive or grading actions.
            """)
            with gr.Row():
                if RESIDUAL_PLOT_PATH.exists():
                    gr.Image(value=str(RESIDUAL_PLOT_PATH), label="Residual Diagnostics")
                if IMPORTANCE_PLOT_PATH.exists():
                    gr.Image(value=str(IMPORTANCE_PLOT_PATH), label="Permutation Feature Importance")


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    )
