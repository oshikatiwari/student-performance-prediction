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

# Load trained pipeline
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
        return 0.0, "Model not loaded", "0% – 0%", "Model unavailable."

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
    range_str = f"{ci_lower}% – {ci_upper}%"

    if score >= 75.0:
        tier = "🌟 Distinction"
        status = f"✅ Student is projected for **High Distinction** ({score:.1f} / 100)"
    elif score >= 50.0:
        tier = "✅ Passing"
        status = f"ℹ️ Student is on track for a **Satisfactory Pass** ({score:.1f} / 100)"
    else:
        tier = "⚠️ At-Risk"
        status = f"🚨 Student is currently **At-Risk of Failure** ({score:.1f} / 100)"

    return round(score, 1), tier, range_str, status


def predict_batch(csv_file):
    if pipeline is None or csv_file is None:
        return gr.update(visible=False), gr.update(visible=False)

    file_path = csv_file.name if hasattr(csv_file, "name") else csv_file
    if not file_path or not Path(file_path).exists():
        return gr.update(visible=False), gr.update(visible=False)

    df_in = pd.read_csv(file_path)
    ids = df_in["ID"] if "ID" in df_in.columns else pd.Series(range(100001, 100001 + len(df_in)), name="ID")

    preds = np.clip(pipeline.predict(df_in), 0.0, 100.0)
    out_df = pd.DataFrame({
        "ID": ids,
        "FinalExamScore": np.round(preds, 2)
    })

    out_csv = ROOT / "submission.csv"
    out_df.to_csv(out_csv, index=False)

    return gr.update(value=out_df.head(10), visible=True), gr.update(value=str(out_csv), visible=True)


# ---------------------------------------------------------
# Clean Gradio Interface
# ---------------------------------------------------------
with gr.Blocks(title="Student Performance Predictor") as demo:
    gr.Markdown("# 🎓 Student Performance Predictor")
    gr.Markdown("Enter student details and click **Predict!** to forecast final examination score and academic standing.")

    with gr.Tabs():
        with gr.TabItem("⚡ Single Prediction"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📚 Academic Background")
                    prev_score = gr.Slider(0.0, 100.0, value=72.0, step=0.5, label="Previous Exam Score")
                    attendance = gr.Slider(0.0, 100.0, value=82.0, step=1.0, label="Attendance Percentage (%)")
                    assignments = gr.Slider(0.0, 100.0, value=85.0, step=1.0, label="Assignments Completed (%)")
                    backlogs = gr.Number(value=0, precision=0, label="Previous Pending Backlogs")

                with gr.Column():
                    gr.Markdown("### 🕒 Study & Daily Habits")
                    study_hours = gr.Slider(0.0, 16.0, value=5.0, step=0.5, label="Daily Study Hours")
                    sleep_hours = gr.Slider(3.0, 12.0, value=7.0, step=0.5, label="Daily Sleep Hours")
                    participation = gr.Slider(0.0, 10.0, value=6.5, step=0.5, label="Class Participation (0–10)")
                    extracurricular = gr.Slider(0.0, 20.0, value=4.0, step=0.5, label="Weekly Extracurricular Hours")

            predict_btn = gr.Button("Predict!", variant="primary")

            with gr.Row():
                score_out = gr.Number(label="Predicted Score (/ 100)")
                tier_out = gr.Textbox(label="Academic Standing")
                range_out = gr.Textbox(label="Estimated Range")

            status_out = gr.Markdown()

            predict_btn.click(
                fn=predict_student,
                inputs=[
                    prev_score, attendance, assignments, backlogs,
                    study_hours, sleep_hours, participation, extracurricular
                ],
                outputs=[score_out, tier_out, range_out, status_out]
            )

        with gr.TabItem("📁 Batch CSV Prediction"):
            gr.Markdown("Upload a student CSV file to generate predictions matching the required `ID,FinalExamScore` format.")
            file_in = gr.File(label="Upload Student CSV", file_types=[".csv"])
            batch_btn = gr.Button("Generate Predictions", variant="primary")

            preview_table = gr.Dataframe(label="Predictions Preview (Top 10)", visible=False)
            download_csv = gr.File(label="Download Generated submission.csv", visible=False)

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
