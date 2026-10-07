"""
Student Performance Predictor — Gradio Interface
Clean, minimalist, and user-friendly prediction application.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import gradio as gr

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "model_pipeline.joblib"
DEFAULT_TEST_CSV = ROOT / "data" / "student_performance_test.csv"

RMSE_ESTIMATE = 6.675
CLASS_MEDIAN = 73.80

# Load trained model pipeline
if MODEL_PATH.exists():
    pipeline = joblib.load(MODEL_PATH)
else:
    pipeline = None


def predict_student(
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
        return 0.0, "Model not available", "[0.0, 0.0]", "Error: Model artifact not found."

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
        tier = "🌟 Distinction (Top Tier)"
        status = f"✅ **High Academic Standing** — Projected: **{score:.2f} / 100** ({delta_str} vs class median)"
    elif score >= 50.0:
        tier = "✅ Passing / Satisfactory"
        status = f"ℹ️ **Satisfactory Standing** — Projected: **{score:.2f} / 100** ({delta_str} vs class median)"
    else:
        tier = "⚠️ At-Risk of Failure"
        status = f"🚨 **At-Risk Alert** — Projected: **{score:.2f} / 100** ({delta_str} vs class median)"

    return round(score, 2), tier, interval_str, status


def predict_batch(csv_file):
    if pipeline is None:
        return None, None

    file_path = csv_file.name if (csv_file and hasattr(csv_file, "name")) else (DEFAULT_TEST_CSV if DEFAULT_TEST_CSV.exists() else None)
    if not file_path or not Path(file_path).exists():
        return None, None

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


# ---------------------------------------------------------
# Clean Gradio Layout
# ---------------------------------------------------------
with gr.Blocks(title="Student Performance Predictor") as demo:
    gr.Markdown("# 🎓 Student Performance Predictor")
    gr.Markdown("Enter student details and click **Predict Exam Score** to forecast final examination performance.")

    with gr.Tabs():
        # TAB 1: Single Prediction
        with gr.TabItem("⚡ Single Prediction"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📚 Academic Metrics")
                    prev_score = gr.Slider(0.0, 100.0, value=72.0, step=0.5, label="Previous Exam Score (0–100)")
                    attendance = gr.Slider(0.0, 100.0, value=82.0, step=1.0, label="Attendance Percentage (%)")
                    assignments = gr.Slider(0.0, 100.0, value=85.0, step=1.0, label="Assignments Completed (%)")
                    backlogs = gr.Number(value=0, precision=0, label="Previous Pending Backlogs")

                with gr.Column():
                    gr.Markdown("### 🕒 Study & Lifestyle Metrics")
                    study_hours = gr.Slider(0.0, 16.0, value=5.0, step=0.5, label="Daily Study Hours")
                    sleep_hours = gr.Slider(3.0, 12.0, value=7.0, step=0.5, label="Daily Sleep Hours")
                    participation = gr.Slider(0.0, 10.0, value=6.5, step=0.5, label="Class Participation (0–10)")
                    extracurricular = gr.Slider(0.0, 20.0, value=4.0, step=0.5, label="Weekly Extracurricular Hours")

            predict_btn = gr.Button("🚀 Predict Exam Score", variant="primary")

            with gr.Row():
                score_out = gr.Number(label="Predicted Final Score (/ 100)", precision=2)
                tier_out = gr.Textbox(label="Academic Standing Tier")
                ci_out = gr.Textbox(label="95% Prediction Interval")

            status_out = gr.Markdown()

            predict_btn.click(
                fn=predict_student,
                inputs=[
                    prev_score, attendance, assignments, backlogs,
                    study_hours, sleep_hours, participation, extracurricular
                ],
                outputs=[score_out, tier_out, ci_out, status_out]
            )

        # TAB 2: Batch CSV Scoring
        with gr.TabItem("📁 Batch CSV Scoring"):
            gr.Markdown("Upload any CSV containing student features to generate batch predictions matching official `ID,FinalExamScore` format.")
            file_in = gr.File(label="Upload Student CSV", file_types=[".csv"])
            batch_btn = gr.Button("⚡ Generate Predictions", variant="primary")

            with gr.Row():
                preview_table = gr.Dataframe(label="Predictions Preview (First 10 Rows)")
                download_csv = gr.File(label="Download Generated submission.csv")

            batch_btn.click(
                fn=predict_batch,
                inputs=[file_in],
                outputs=[preview_table, download_csv]
            )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    )
