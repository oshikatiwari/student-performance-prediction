# GDG NMIT — Student Performance Prediction (Round 2)

Supervised Machine Learning Regression Pipeline for academic exam performance prediction (`FinalExamScore`).

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_Demo-red.svg)](https://streamlit.io)

---

## 📌 Project Overview
Developed for **GDG NMIT Recruitments 5.0 (Machine Learning Domain)**.

- **Objective:** Predict university student `FinalExamScore` from academic and behavioral indicators.
- **Key Focus Areas:**
  - Zero-leakage Scikit-Learn `Pipeline` / `ColumnTransformer`.
  - Comprehensive data audit & pre-exam feature justification (`PostExamConfidence` identified as critical leakage and dropped).
  - 5-Fold Cross-Validation benchmark across 5 distinct model families reporting **mean ± std**.
  - Systematic residual error diagnostics & failing segment root cause analysis.
  - Runnable inference CLI (`predict.py`) and interactive Streamlit web application.
  - Comprehensive half-page **Model Card** documented in `REPORT.md`.

---

## 🗂️ Repository Structure
```
├── data/
│   ├── student_performance.csv       # Training dataset (1000 rows, 11 cols)
│   └── student_performance_test.csv  # Official test set (200 rows)
├── artifacts/
│   ├── model_pipeline.joblib         # Fitted, serialized scikit-learn pipeline
│   ├── metrics_summary.json          # 5-fold CV benchmarking metrics
│   ├── residual_plot.png             # Out-of-fold diagnostic residual plots
│   └── feature_importance.png        # Permutation feature importance chart
├── src/
│   ├── __init__.py
│   ├── audit.py                      # Data audit, sentinel check, & leakage justification
│   ├── pipeline.py                   # Atomic scikit-learn preprocessing & model pipeline
│   ├── train.py                      # 5-fold CV model benchmarking & hyperparameter tuning
│   ├── evaluate.py                   # Out-of-fold residual diagnostics & segment error analysis
│   └── explain.py                    # Permutation feature importance analysis
├── predict.py                        # Packaged CLI: --input X.csv --output submission.csv
├── app.py                            # Interactive Streamlit application
├── submission.csv                    # Final predictions on test dataset (ID,FinalExamScore)
├── requirements.txt                  # Pinned dependencies
├── REPORT.md                         # Full technical report, audit, comparison table & model card
└── README.md                         # Quickstart documentation
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
Clone the repository and install required packages:
```bash
git clone <repo-url>
cd gdg-nmit-ml-round2
pip install -r requirements.txt
```

### 2. Run Data Audit & Leakage Check
```bash
python -m src.audit
```

### 3. Train & Benchmark All Models (5-Fold Cross-Validation)
```bash
python -m src.train
```

### 4. Residual Diagnostics & Explainability
```bash
python -m src.evaluate
python -m src.explain
```

### 5. Packaged CLI Batch Inference
Generate predictions for any student test CSV file matching the required `ID,FinalExamScore` output format:
```bash
python predict.py --input data/student_performance_test.csv --output submission.csv
```

### 6. Launch Interactive Streamlit App
```bash
streamlit run app.py
```

---

## 📊 5-Fold Cross-Validation Benchmark Summary

| Model Family | CV RMSE (Mean ± Std) | CV MAE (Mean ± Std) | CV $R^2$ (Mean ± Std) |
| :--- | :---: | :---: | :---: |
| 1. Mean Baseline | $13.989 \pm 1.194$ | $11.126 \pm 0.937$ | $-0.010 \pm 0.010$ |
| 2. Ridge Regression | $7.945 \pm 0.621$ | $5.960 \pm 0.352$ | $0.671 \pm 0.044$ |
| 3. Random Forest Regressor | $7.429 \pm 0.806$ | $5.714 \pm 0.550$ | $0.713 \pm 0.048$ |
| 4. Hist Gradient Boosting | $7.168 \pm 0.739$ | $5.563 \pm 0.504$ | $0.734 \pm 0.025$ |
| **5. Tuned Gradient Boosting** | **$6.675 \pm 0.494$** | **$5.276 \pm 0.376$** | **$0.764 \pm 0.021$** |

*Detailed methodology, error analysis, and Model Card are available in [`REPORT.md`](REPORT.md).*
