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

RMSE_ESTIMATE = 6.675
CLASS_MEDIAN = 73.80


@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        st.error("Model pipeline artifact not found. Please train the model first.")
        return None
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🎓 Student Performance Predictor")
st.write(
    "Enter student details and click **Predict!** to forecast their final examination score, "
    "readiness range, and academic standing tier."
)

st.divider()

# ---------------------------------------------------------
# Input Features
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📚 Academic Background")
    previous_score = st.slider("Previous Exam Score", min_value=0.0, max_value=100.0, value=72.0, step=0.5)
    attendance = st.slider("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=82.0, step=1.0)
    assignments = st.slider("Assignments Completed (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
    backlogs = st.number_input("Previous Pending Backlogs", min_value=0, max_value=10, value=0, step=1)

with col2:
    st.markdown("#### 🕒 Study & Daily Habits")
    study_hours = st.slider("Daily Study Hours", min_value=0.0, max_value=16.0, value=5.0, step=0.5)
    sleep_hours = st.slider("Daily Sleep Hours", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
    participation = st.slider("Class Participation (0–10)", min_value=0.0, max_value=10.0, value=6.5, step=0.5)
    extracurricular = st.slider("Weekly Extracurricular Hours", min_value=0.0, max_value=20.0, value=4.0, step=0.5)

st.caption("Tip: Higher previous exam scores and fewer backlogs strongly increase the expected final score.")
st.divider()

# ---------------------------------------------------------
# Prediction Action & Results
# ---------------------------------------------------------
predict_button = st.button("Predict!", type="primary")

if predict_button and pipeline is not None:
    features = pd.DataFrame([{
        "StudyHours": study_hours,
        "AttendancePercentage": attendance,
        "PreviousExamScore": previous_score,
        "AssignmentsCompleted": assignments,
        "SleepHours": sleep_hours,
        "ExtracurricularHours": extracurricular,
        "ClassParticipation": participation,
        "PreviousBacklogs": backlogs
    }])

    raw_prediction = pipeline.predict(features)[0]
    score = float(np.clip(raw_prediction, 0.0, 100.0))

    ci_lower = max(0.0, round(score - 1.96 * RMSE_ESTIMATE, 1))
    ci_upper = min(100.0, round(score + 1.96 * RMSE_ESTIMATE, 1))

    # Clean outcome banner
    if score >= 75.0:
        st.success(f"Student is projected to achieve **HIGH DISTINCTION** ({score:.1f} / 100)")
        tier = "Distinction"
    elif score >= 50.0:
        st.info(f"Student is on track for a **SATISFACTORY PASS** ({score:.1f} / 100)")
        tier = "Passing"
    else:
        st.error(f"Student is currently **AT RISK OF DEFICIENCY** ({score:.1f} / 100)")
        tier = "At-Risk"

    st.subheader("Performance Breakdown")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Score", f"{score:.1f} / 100")
    m2.metric("Estimated Range", f"{ci_lower}% – {ci_upper}%")
    m3.metric("Academic Standing", tier)

    st.progress(score / 100.0, text=f"Readiness Score: {score:.1f}%")

    st.markdown("##### Score Comparison")
    st.bar_chart(
        pd.DataFrame(
            {"Score": [40.0, CLASS_MEDIAN, score]},
            index=["Passing Threshold", "Class Median", "Predicted Student"]
        )
    )
else:
    st.info("Enter values above, then click **Predict!**")

# ---------------------------------------------------------
# Optional Batch Prediction (Clean Accordion)
# ---------------------------------------------------------
st.divider()
with st.expander("📁 Batch Prediction (Upload CSV)"):
    st.write("Upload a student CSV file to generate predictions matching the required `ID,FinalExamScore` format.")
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None and pipeline is not None:
        batch_df = pd.read_csv(uploaded_file)
        ids = batch_df["ID"] if "ID" in batch_df.columns else pd.Series(range(100001, 100001 + len(batch_df)), name="ID")
        preds = np.clip(pipeline.predict(batch_df), 0.0, 100.0)

        result_df = pd.DataFrame({
            "ID": ids,
            "FinalExamScore": np.round(preds, 2)
        })

        st.dataframe(result_df.head(10))

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download submission.csv",
            data=csv_bytes,
            file_name="submission.csv",
            mime="text/csv",
            type="primary"
        )
