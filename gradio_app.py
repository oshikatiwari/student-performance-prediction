"""
EduPredict Pro — Student Performance Prediction System (Gradio Edition)
GDG NMIT - Machine Learning Technical Round 2
Inspired by modern full-stack academic advisory platforms.
Features:
- Student What-If Habit Simulator & Actionable Academic Coaching
- Faculty / Advisor Cohort Screener & At-Risk Early Warning System
- Official Test Batch Inference & submission.csv Generator
- Viva Defense & Empirical Validation Architecture
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
DEFAULT_TEST_CSV = ROOT / "data" / "student_performance_test.csv"
DEFAULT_TRAIN_CSV = ROOT / "data" / "student_performance.csv"

RMSE_ESTIMATE = 6.675
CLASS_MEDIAN = 73.80

# Load trained pipeline
if MODEL_PATH.exists():
    pipeline = joblib.load(MODEL_PATH)
else:
    pipeline = None


# -----------------------------------------------------------------------------
# Core Prediction & Simulation Logic
# -----------------------------------------------------------------------------
def predict_and_simulate(
    previous_score: float,
    attendance: float,
    assignments: float,
    backlogs: float,
    study_hours: float,
    sleep_hours: float,
    participation: float,
    extracurricular: float,
    target_extra_study: float
):
    if pipeline is None:
        return 0.0, "Model not loaded", "[0.0, 0.0]", "Error: Model artifact not found.", ""

    # 1. Base prediction
    base_df = pd.DataFrame([{
        "StudyHours": study_hours,
        "AttendancePercentage": attendance,
        "PreviousExamScore": previous_score,
        "AssignmentsCompleted": assignments,
        "SleepHours": sleep_hours,
        "ExtracurricularHours": extracurricular,
        "ClassParticipation": participation,
        "PreviousBacklogs": backlogs
    }])

    raw_pred = pipeline.predict(base_df)[0]
    score = float(np.clip(raw_pred, 0.0, 100.0))

    ci_lower = max(0.0, round(score - 1.96 * RMSE_ESTIMATE, 1))
    ci_upper = min(100.0, round(score + 1.96 * RMSE_ESTIMATE, 1))
    interval_str = f"[{ci_lower} , {ci_upper}]"

    delta = score - CLASS_MEDIAN
    delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"

    if score >= 75.0:
        tier = "🌟 Distinction (Top Tier)"
        status_box = (
            f"### ✅ **High Academic Standing**\n\n"
            f"- **Predicted Final Exam Score:** `{score:.2f} / 100` ({delta_str} pts vs class median)\n"
            f"- **95% Confidence Bounds:** `{interval_str}`\n"
            f"- **Standing:** Student is well-prepared for distinction honors."
        )
    elif score >= 50.0:
        tier = "✅ Passing / Satisfactory"
        status_box = (
            f"### ℹ️ **Satisfactory Academic Standing**\n\n"
            f"- **Predicted Final Exam Score:** `{score:.2f} / 100` ({delta_str} pts vs class median)\n"
            f"- **95% Confidence Bounds:** `{interval_str}`\n"
            f"- **Standing:** On track to pass. Proactive study adjustments can push into distinction."
        )
    else:
        tier = "⚠️ At-Risk of Failure"
        status_box = (
            f"### 🚨 **At-Risk Warning (Intervention Required)**\n\n"
            f"- **Predicted Final Exam Score:** `{score:.2f} / 100` ({delta_str} pts vs class median)\n"
            f"- **95% Confidence Bounds:** `{interval_str}`\n"
            f"- **Standing:** High probability of academic deficiency. Early mentorship strongly recommended."
        )

    # 2. What-If Simulation: Increasing study hours & optimal sleep
    sim_df = pd.DataFrame([{
        "StudyHours": min(16.0, study_hours + target_extra_study),
        "AttendancePercentage": min(100.0, attendance + 5.0),
        "PreviousExamScore": previous_score,
        "AssignmentsCompleted": min(100.0, assignments + 5.0),
        "SleepHours": 7.5 if sleep_hours < 6.0 else sleep_hours,
        "ExtracurricularHours": extracurricular,
        "ClassParticipation": min(10.0, participation + 1.0),
        "PreviousBacklogs": backlogs
    }])

    sim_raw = pipeline.predict(sim_df)[0]
    sim_score = float(np.clip(sim_raw, 0.0, 100.0))
    score_gain = sim_score - score

    simulation_text = (
        f"#### 📈 **What-If Habit Impact Forecast**\n\n"
        f"If the student dedicates **+{target_extra_study:.1f} hours/day** to self-study, ensures adequate sleep (7.5h), and boosts attendance by +5%:\n\n"
        f"- **Projected New Score:** `{sim_score:.2f} / 100`  \n"
        f"- **Estimated Score Boost:** **+{score_gain:+.2f} points**  \n"
        f"- **Trajectory Shift:** {tier} ➔ **{'🌟 Distinction' if sim_score >= 75 else ('✅ Passing' if sim_score >= 50 else '⚠️ At-Risk')}**"
    )

    return round(score, 2), tier, interval_str, status_box, simulation_text


# -----------------------------------------------------------------------------
# Faculty / Advisor Cohort Screener Logic
# -----------------------------------------------------------------------------
def analyze_cohort(csv_file):
    if pipeline is None:
        return "Model not available", None, None

    file_path = csv_file.name if (csv_file and hasattr(csv_file, "name")) else (DEFAULT_TRAIN_CSV if DEFAULT_TRAIN_CSV.exists() else None)
    if file_path is None or not os.path.exists(file_path):
        return "No dataset found.", None, None

    df_in = pd.read_csv(file_path)
    preds = np.clip(pipeline.predict(df_in), 0.0, 100.0)
    df_eval = df_in.copy()
    df_eval["PredictedScore"] = np.round(preds, 2)

    total_students = len(df_eval)
    at_risk = df_eval[df_eval["PredictedScore"] < 50.0]
    distinction = df_eval[df_eval["PredictedScore"] >= 75.0]
    avg_score = df_eval["PredictedScore"].mean()

    summary_md = (
        f"### 🏫 **Cohort Analysis Summary ({total_students} Students)**\n\n"
        f"- **Cohort Projected Average:** `{avg_score:.2f} / 100`\n"
        f"- **Students on Track for Distinction (>= 75):** `{len(distinction)} ({len(distinction)/total_students*100:.1f}%)`\n"
        f"- **Students Flagged At-Risk (< 50):** **`{len(at_risk)} ({len(at_risk)/total_students*100:.1f}%)`**\n"
        f"- **Actionable Directive:** Review the at-risk student table below for scheduled remedial coaching."
    )

    show_cols = [c for c in ["ID", "PreviousBacklogs", "AttendancePercentage", "PreviousExamScore", "StudyHours", "PredictedScore"] if c in df_eval.columns]
    at_risk_table = at_risk[show_cols].sort_values("PredictedScore").head(15)
    top_table = distinction[show_cols].sort_values("PredictedScore", ascending=False).head(10)

    return summary_md, at_risk_table, top_table


# -----------------------------------------------------------------------------
# Batch Scoring Logic
# -----------------------------------------------------------------------------
def process_batch_file(csv_file):
    if pipeline is None:
        return None, None

    file_path = csv_file.name if (csv_file and hasattr(csv_file, "name")) else (DEFAULT_TEST_CSV if DEFAULT_TEST_CSV.exists() else None)
    if not file_path or not os.path.exists(file_path):
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


# -----------------------------------------------------------------------------
# UI Construction
# -----------------------------------------------------------------------------
custom_css = """
footer {visibility: hidden}
.gradio-container {max-width: 1200px !important}
"""

with gr.Blocks(title="EduPredict — Student Performance Prediction System") as demo:
    gr.Markdown("""
    # 🎓 **EduPredict — Student Performance Prediction System**
    *Supervised Machine Learning Pipeline for Prospective Academic Advisory & Risk Prevention*
    """)

    with gr.Tabs():
        # TAB 1: STUDENT PORTAL & WHAT-IF SIMULATOR
        with gr.TabItem("🎯 Student Portal & What-If Planner"):
            gr.Markdown("Estimate your expected final exam score and simulate how targeted habit changes elevate your standing.")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📚 Academic Track Record")
                    prev_score_in = gr.Slider(0.0, 100.0, value=72.0, step=0.5, label="Previous Exam Score (0–100)")
                    attendance_in = gr.Slider(0.0, 100.0, value=82.0, step=1.0, label="Attendance Percentage (%)")
                    assign_in = gr.Slider(0.0, 100.0, value=85.0, step=1.0, label="Assignments Completed (%)")
                    backlogs_in = gr.Number(value=0, precision=0, label="Historical Course Backlogs")

                with gr.Column():
                    gr.Markdown("#### 🕒 Preparation & Engagement")
                    study_in = gr.Slider(0.0, 16.0, value=5.0, step=0.5, label="Daily Study Hours")
                    sleep_in = gr.Slider(3.0, 12.0, value=7.0, step=0.5, label="Daily Sleep Duration (Hours)")
                    particip_in = gr.Slider(0.0, 10.0, value=6.5, step=0.5, label="Class Participation Score (0–10)")
                    extra_in = gr.Slider(0.0, 20.0, value=4.0, step=0.5, label="Weekly Extracurricular Hours")

            with gr.Row():
                extra_study_slider = gr.Slider(0.5, 5.0, value=2.0, step=0.5, label="⚡ 'What-If' Habit Simulation: Additional Daily Study Hours")

            calc_btn = gr.Button("🚀 Calculate Predicted Exam Score & Habit Impact", variant="primary")

            with gr.Row():
                score_box = gr.Number(label="Predicted Score (/ 100)", precision=2)
                tier_box = gr.Textbox(label="Academic Standing Category")
                ci_box = gr.Textbox(label="95% Prediction Interval")

            with gr.Row():
                status_md = gr.Markdown()
                simulation_md = gr.Markdown()

            calc_btn.click(
                fn=predict_and_simulate,
                inputs=[
                    prev_score_in, attendance_in, assign_in, backlogs_in,
                    study_in, sleep_in, particip_in, extra_in, extra_study_slider
                ],
                outputs=[score_box, tier_box, ci_box, status_md, simulation_md]
            )

        # TAB 2: FACULTY / ADVISOR COHORT SCREENER
        with gr.TabItem("👨‍🏫 Faculty / Advisor Screener"):
            gr.Markdown("Screen classroom cohorts to proactively identify at-risk students who need academic counseling before final examinations.")
            with gr.Row():
                cohort_file_input = gr.File(label="Upload Class Cohort CSV (Leave empty to audit training dataset)", file_types=[".csv"])
            screen_btn = gr.Button("🔍 Run Cohort Screening", variant="primary")

            cohort_summary = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🚨 Priority At-Risk Students (Predicted Score < 50)")
                    at_risk_df = gr.Dataframe(label="At-Risk Student Flag List")
                with gr.Column():
                    gr.Markdown("#### 🌟 Top Performers (Predicted Score ≥ 75)")
                    top_df = gr.Dataframe(label="Distinction Honor Roll")

            screen_btn.click(
                fn=analyze_cohort,
                inputs=[cohort_file_input],
                outputs=[cohort_summary, at_risk_df, top_df]
            )

        # TAB 3: BATCH CSV INFERENCE (SUBMISSION GENERATOR)
        with gr.TabItem("📁 Batch CSV Scoring"):
            gr.Markdown("Score unseen student test datasets and generate the official `submission.csv` strictly matching `ID,FinalExamScore`.")
            batch_upload = gr.File(label="Upload Test CSV (Defaults to data/student_performance_test.csv)", file_types=[".csv"])
            batch_btn = gr.Button("⚡ Generate Test Predictions", variant="primary")

            with gr.Row():
                batch_preview = gr.Dataframe(label="Preview Top 10 Predictions")
                batch_download = gr.File(label="Download Generated submission.csv")

            batch_btn.click(
                fn=process_batch_file,
                inputs=[batch_upload],
                outputs=[batch_preview, batch_download]
            )

        # TAB 4: VIVA DEFENSE & MODEL CARD
        with gr.TabItem("📊 Viva Defense & Model Diagnostics"):
            gr.Markdown(r"""
            ### 💡 **Interview & Viva Defense Guide**
            
            **Q: Why is your Cross-Validation R² ~0.764 and not >0.95?**  
            *Defense:* Real behavioral and academic data depends on unmeasured human factors (e.g. subject difficulty, physical health on exam day, test anxiety). An $R^2$ of $\approx 0.764$ explains **over 76% of exam variance** honestly. On real behavioral data, any model claiming $R^2 > 0.95$ almost certainly suffers from **data leakage or severe overfitting**, which would fail on unseen cohorts.

            **Q: How did you ensure zero data leakage?**  
            *Defense:* 
            1. Identified and dropped **`PostExamConfidence`**: soliciting confidence *after* the exam leaks post-event signals that cannot exist in prospective advising.
            2. All preprocessing (imputation, scaling, bounds sanitization) is enclosed inside an atomic Scikit-Learn `Pipeline` fitted **strictly on training folds** inside 5-Fold Cross-Validation.

            **Q: What is the primary underperforming student segment?**  
            *Defense:* Students with $\ge 3$ historical backlogs (MAE = 6.50 vs 4.75 for 0 backlogs). The model systematically overpredicts their score because backlogs carry compounded psychological test stress across multiple remedial exams that study hours alone cannot account for.
            """)

            gr.Markdown("#### 📈 5-Fold Cross-Validation Benchmark")
            gr.Markdown(r"""
            | Model Family | CV RMSE (Mean ± Std) | CV MAE (Mean ± Std) | CV $R^2$ (Mean ± Std) |
            | :--- | :---: | :---: | :---: |
            | **1. Mean Baseline** | $13.989 \pm 1.194$ | $11.126 \pm 0.937$ | $-0.010 \pm 0.010$ |
            | **2. Ridge Regression** | $7.945 \pm 0.621$ | $5.960 \pm 0.352$ | $0.671 \pm 0.044$ |
            | **3. Random Forest** | $7.429 \pm 0.806$ | $5.714 \pm 0.550$ | $0.713 \pm 0.048$ |
            | **4. HistGradientBoosting** | $7.168 \pm 0.739$ | $5.563 \pm 0.504$ | $0.734 \pm 0.025$ |
            | **5. Tuned Gradient Boosting** | **$6.675 \pm 0.494$** | **$5.276 \pm 0.376$** | **$0.764 \pm 0.021$** |
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
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")
    )
