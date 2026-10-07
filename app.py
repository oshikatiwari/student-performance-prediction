import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "model_pipeline.joblib"
RESIDUAL_PLOT_PATH = ROOT / "artifacts" / "residual_plot.png"
IMPORTANCE_PLOT_PATH = ROOT / "artifacts" / "feature_importance.png"

RMSE_ESTIMATE = 6.675  # 5-Fold CV RMSE from tuned Gradient Boosting pipeline
CLASS_MEDIAN = 73.80   # Benchmark training class median score


# ---------------------------------------------------------
# Load Saved Pipeline Artifact
# ---------------------------------------------------------
@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        st.error(f"Model artifact not found at `{MODEL_PATH}`. Please run `python -m src.train` first.")
        return None
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()

# ---------------------------------------------------------
# Header & Navigation
# ---------------------------------------------------------
st.title("🎓 Student Performance Predictor")
st.write(
    "Predict student final exam scores and academic standing using verified pre-exam indicators."
)

tab1, tab2 = st.tabs(["⚡ Single Prediction", "📁 Batch CSV Scoring"])

# ---------------------------------------------------------
# Tab 1: Single Prediction
# ---------------------------------------------------------
with tab1:
    st.subheader("Student Academic & Preparation Profile")
    st.write("Adjust the parameters below to evaluate predicted exam performance and risk status.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📚 Academic Track Record")
        previous_score = st.slider("Previous Exam Score (0–100)", min_value=0.0, max_value=100.0, value=72.0, step=0.5)
        attendance = st.slider("Classroom Attendance (%)", min_value=0.0, max_value=100.0, value=82.0, step=1.0)
        assignments = st.slider("Assignments Completed (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
        backlogs = st.number_input("Previous Pending Backlogs", min_value=0, max_value=10, value=0, step=1)

    with col2:
        st.markdown("#### 🕒 Preparation & Engagement")
        study_hours = st.slider("Daily Self-Study Hours", min_value=0.0, max_value=16.0, value=5.0, step=0.5)
        sleep_hours = st.slider("Daily Sleep Duration (Hours)", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
        participation = st.slider("Class Participation Score (0–10)", min_value=0.0, max_value=10.0, value=6.5, step=0.5)
        extracurricular = st.slider("Weekly Extracurricular Hours", min_value=0.0, max_value=20.0, value=4.0, step=0.5)

    st.caption("Tip: Previous exam score and backlog count have the strongest predictive influence on the final outcome.")
    st.divider()

    predict_btn = st.button("Predict Exam Score", type="primary")

    if predict_btn and pipeline is not None:
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

        # Outcome Status Card
        if score >= 75.0:
            st.success(f"🌟 **High Academic Distinction** — Projected Score: **{score:.2f} / 100**")
        elif score >= 50.0:
            st.info(f"✅ **Satisfactory Standing (Passing)** — Projected Score: **{score:.2f} / 100**")
        else:
            st.error(f"⚠️ **At-Risk Student (Potential Failure)** — Projected Score: **{score:.2f} / 100**")

        st.subheader("Performance Metrics & Uncertainty")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Score", f"{score:.2f} / 100", delta=f"{score - CLASS_MEDIAN:+.1f} vs class median")
        m2.metric("95% Prediction Interval", f"[{ci_lower} , {ci_upper}]", help="Uncertainty range based on 5-fold cross-validated RMSE")
        
        tier_label = "Distinction" if score >= 75 else ("Pass" if score >= 50 else "At-Risk")
        m3.metric("Performance Tier", tier_label)

        st.progress(score / 100.0, text=f"Exam Readiness: {score:.1f}%")

        # Visual Benchmark Chart
        st.markdown("##### Benchmark Comparison")
        chart_data = pd.DataFrame({
            "Score": [40.0, CLASS_MEDIAN, score]
        }, index=["Pass Threshold", "Cohort Median", "Predicted Student"])
        st.bar_chart(chart_data)

    else:
        st.info("Adjust the parameters above and click **Predict Exam Score** to view the forecast.")


# ---------------------------------------------------------
# Tab 2: Batch CSV Scoring
# ---------------------------------------------------------
with tab2:
    st.subheader("Batch CSV Prediction")
    st.write("Upload a CSV file containing student features to generate batch predictions matching official submission format (`ID,FinalExamScore`).")

    uploaded_file = st.file_uploader("Upload Student Features CSV", type=["csv"])

    if uploaded_file is not None and pipeline is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(batch_df)}** records with columns: `{', '.join(batch_df.columns)}`")

        if st.button("Generate Batch Predictions", type="primary"):
            if "ID" in batch_df.columns:
                ids = batch_df["ID"]
            else:
                ids = pd.Series(range(100001, 100001 + len(batch_df)), name="ID")

            preds = np.clip(pipeline.predict(batch_df), 0.0, 100.0)
            result_df = pd.DataFrame({
                "ID": ids,
                "FinalExamScore": np.round(preds, 2)
            })

            st.dataframe(result_df.head(10), use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download submission.csv",
                data=csv_data,
                file_name="submission.csv",
                mime="text/csv",
                type="primary"
            )
