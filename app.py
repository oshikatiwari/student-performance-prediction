"""
Interactive Streamlit Application for Student Performance Prediction.
GDG NMIT - Machine Learning Technical Round 2
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "model_pipeline.joblib")
RMSE_ESTIMATE = 6.68  # 5-Fold CV RMSE from tuned model for prediction interval


st.set_page_config(
    page_title="Student Performance Predictor | GDG NMIT",
    page_icon="🎓",
    layout="wide"
)

# Custom header styling
st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a, #3b82f6); padding: 24px; border-radius: 12px; color: white; margin-bottom: 24px;">
        <h1 style="margin: 0; font-size: 2.2rem;">🎓 Student Performance Predictor</h1>
        <p style="margin: 8px 0 0 0; font-size: 1.05rem; opacity: 0.9;">
            GDG NMIT Recruitment 5.0 — Machine Learning Domain Challenge
        </p>
    </div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model artifact not found at {MODEL_PATH}. Run 'python -m src.train' first.")
        return None
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()

tabs = st.tabs(["🎯 Single Student Prediction", "📁 Batch CSV Scoring", "📊 Model Insights & Card"])

with tabs[0]:
    st.subheader("Predict Student Final Exam Score")
    st.markdown("Enter pre-exam academic and lifestyle metrics to estimate the expected `FinalExamScore` (0-100 scale).")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📚 Academic Metrics")
        previous_score = st.slider("Previous Exam Score (0 - 100)", min_value=0.0, max_value=100.0, value=72.0, step=0.5)
        attendance = st.slider("Classroom Attendance (%)", min_value=0.0, max_value=100.0, value=82.0, step=1.0)
        assignments = st.slider("Assignments Completed (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
        backlogs = st.number_input("Previous Pending Backlogs", min_value=0, max_value=10, value=0, step=1)

    with col2:
        st.markdown("##### 🕒 Study & Lifestyle Metrics")
        study_hours = st.slider("Daily Study Hours", min_value=0.0, max_value=16.0, value=5.0, step=0.5)
        sleep_hours = st.slider("Daily Sleep Hours", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
        participation = st.slider("Class Participation Score (0 - 10)", min_value=0.0, max_value=10.0, value=6.5, step=0.5)
        extracurricular = st.slider("Weekly Extracurricular Hours", min_value=0.0, max_value=20.0, value=4.0, step=0.5)

    if st.button("🚀 Calculate Predicted Score", type="primary", use_container_width=True):
        if pipeline is not None:
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

            pred_raw = pipeline.predict(input_df)[0]
            pred_score = float(np.clip(pred_raw, 0.0, 100.0))

            lower_bound = max(0.0, round(pred_score - 1.96 * RMSE_ESTIMATE, 1))
            upper_bound = min(100.0, round(pred_score + 1.96 * RMSE_ESTIMATE, 1))

            st.markdown("---")
            m_col1, m_col2, m_col3 = st.columns(3)

            with m_col1:
                st.metric("Predicted Final Exam Score", f"{pred_score:.2f} / 100")

            with m_col2:
                st.metric("95% Prediction Interval", f"[{lower_bound:.1f} , {upper_bound:.1f}]")

            with m_col3:
                if pred_score >= 80:
                    st.success("🌟 Tier: High Distinction")
                elif pred_score >= 60:
                    st.info("✅ Tier: Satisfactory / Passing")
                elif pred_score >= 45:
                    st.warning("⚠️ Tier: Marginal / Needs Support")
                else:
                    st.error("🚨 Tier: High Risk of Failure")

with tabs[1]:
    st.subheader("Batch Inference via CSV")
    st.markdown("Upload any CSV containing student features to generate `FinalExamScore` predictions matching the official format (`ID,FinalExamScore`).")

    uploaded_file = st.file_uploader("Upload Student Feature CSV", type=["csv"])
    if uploaded_file is not None and pipeline is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(batch_df)} rows. Columns found:", list(batch_df.columns))

        if st.button("Generate Batch Predictions"):
            if "ID" in batch_df.columns:
                ids = batch_df["ID"]
            else:
                ids = pd.Series(range(100001, 100001 + len(batch_df)), name="ID")

            preds = np.clip(pipeline.predict(batch_df), 0.0, 100.0)
            res_df = pd.DataFrame({
                "ID": ids,
                "FinalExamScore": np.round(preds, 2)
            })

            st.dataframe(res_df.head(10), use_container_width=True)
            csv_data = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download submission.csv",
                data=csv_data,
                file_name="submission.csv",
                mime="text/csv",
                type="primary"
            )

with tabs[2]:
    st.subheader("Model Card & Diagnostic Overview")
    st.markdown("""
    - **Architecture**: Scikit-Learn `Pipeline` with `SimpleImputer` (median), `StandardScaler`, and Tuned `GradientBoostingRegressor`.
    - **Cross-Validation**: 5-Fold K-Fold cross validation (Seed = 42).
    - **Top Predictors**: `PreviousExamScore`, `PreviousBacklogs`, `SleepHours`, and `StudyHours`.
    - **Data Leakage Prevention**: `PostExamConfidence` is strictly **excluded** from the inference pipeline because it is recorded after exam completion.
    - **Intended Use**: Academic advising and early support flagging.
    - **Prohibited Use**: High-stakes disciplinary or automated grading decisions.
    """)

    st.markdown("##### Residual Diagnostic and Feature Importance Visualizations")
    img_col1, img_col2 = st.columns(2)
    res_img = os.path.join(os.path.dirname(__file__), "artifacts", "residual_plot.png")
    imp_img = os.path.join(os.path.dirname(__file__), "artifacts", "feature_importance.png")

    if os.path.exists(res_img):
        with img_col1:
            st.image(res_img, caption="Out-of-Fold Residual Diagnostics", use_container_width=True)
    if os.path.exists(imp_img):
        with img_col2:
            st.image(imp_img, caption="Permutation Feature Importance", use_container_width=True)
