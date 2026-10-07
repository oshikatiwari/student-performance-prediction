# GDG NMIT Recruitments 5.0 — Machine Learning Round 2
## Student Performance Prediction Technical Report

**Candidate Track:** Machine Learning  
**Target Column:** `FinalExamScore` (Continuous, scale 0–100)  
**Primary Metric:** Root Mean Squared Error (RMSE)  
**Secondary Metrics:** Mean Absolute Error (MAE), Coefficient of Determination ($R^2$)  
**Validation Strategy:** 5-Fold Cross-Validation (Seed = 42, Shuffle = True)  

---

## Executive Summary
This report presents an end-to-end, production-ready supervised machine learning solution to predict university student final exam scores (`FinalExamScore`). The solution enforces strict prevention of data leakage through an integrated Scikit-Learn `Pipeline`, rigorous cross-validation benchmarking across five candidate algorithms, targeted hyperparameter tuning under a defined compute budget, granular residual error diagnostics, and a packaged CLI and interactive web application.

---

## 1. Comprehensive Data Audit & Leakage Check

### 1.1 Dataset Overview & Missingness
The training dataset (`student_performance.csv`) consists of **1,000 observations** across **11 columns**. The test dataset (`student_performance_test.csv`) consists of **200 observations** across **10 columns** (excluding target).

| Column Name | Dtype | Missing Count (Train) | Missing Pct (Train) | Missing Count (Test) | Missing Pct (Test) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `ID` | `int64` | 0 | 0.0% | 0 | 0.0% |
| `StudyHours` | `float64` | 18 | 1.8% | 2 | 1.0% |
| `AttendancePercentage` | `float64` | 39 | 3.9% | 8 | 4.0% |
| `PreviousExamScore` | `float64` | 32 | 3.2% | 8 | 4.0% |
| `AssignmentsCompleted` | `float64` | 58 | 5.8% | 9 | 4.5% |
| `SleepHours` | `float64` | 44 | 4.4% | 8 | 4.0% |
| `ExtracurricularHours` | `float64` | 34 | 3.4% | 5 | 2.5% |
| `ClassParticipation` | `float64` | 31 | 3.1% | 7 | 3.5% |
| `PreviousBacklogs` | `float64` | 7 | 0.7% | 0 | 0.0% |
| `PostExamConfidence` | `float64` | 44 | 4.4% | 8 | 4.0% |
| `FinalExamScore` (Target) | `float64` | 0 | 0.0% | N/A | N/A |

### 1.2 Target Distribution & Anomalous Records
The target variable `FinalExamScore` displays a roughly normal distribution centered at **73.69**:
- **Mean:** 73.69 | **Median:** 73.80 | **Standard Deviation:** 14.40 | **IQR:** [64.70, 83.00]
- **Target Anomalies (Bad Ground-Truth Records):**
  - Record `ID 100540`: `FinalExamScore = -5.0` (physically impossible score below zero).
  - Record `ID 100746`: `FinalExamScore = 142.5` (physically impossible score exceeding 100).
  - *Action:* Dropped from training folds to prevent severe gradient corruption during model fitting.

### 1.3 Out-of-Bounds & Sentinel Feature Values
Anomalies and sentinel values (e.g. `-1.0` representing missing/unrecorded entries) were identified:
- `StudyHours`: Minimum `-1.0` (sentinel) and maximum `25.0` in train, `99.0` in test (Row ID `101163`).
- `AttendancePercentage`: Maximum `112.0` in train, `105.0` in test (Row ID `101033`), minimum `-1.0` in train.
- `PreviousExamScore`: Maximum `150.0` in train.
- `AssignmentsCompleted`: Maximum `140.0` in train.
- `SleepHours`: Minimum `-1.0` in test (Row ID `101128`), maximum `25.0` in train.
- `PreviousBacklogs`: Minimum `-1.0` in train.

*Pipeline Handling:* To ensure zero test-time inference failures, the custom `SentinelAndBoundsSanitizer` maps all negative sentinel entries (`< 0`) to `NaN` (for median imputation) and clips physical upper bounds (`StudyHours` $\le 24$, `Attendance` $\le 100$, `PreviousScore` $\le 100$, `Assignments` $\le 100$, `Sleep` $\le 24$).

### 1.4 Feature Justification & Pre-Exam Leakage Check

| Feature | Known Pre-Exam? | Decision | Engineering Justification |
| :--- | :---: | :---: | :--- |
| `ID` | No (Arbitrary) | **Drop** | Non-semantic identifier. Dropped to prevent spurious memorization. |
| `StudyHours` | **Yes** | **Keep** | Pre-exam revision and preparation time. Direct behavioral indicator. |
| `AttendancePercentage` | **Yes** | **Keep** | Historical semester classroom presence recorded before finals. |
| `PreviousExamScore` | **Yes** | **Keep** | Pre-final midterm or prerequisite benchmark examination score. |
| `AssignmentsCompleted` | **Yes** | **Keep** | Cumulative percentage of regular coursework completed during term. |
| `SleepHours` | **Yes** | **Keep** | Routine daily rest duration prior to exam period; physical wellbeing signal. |
| `ExtracurricularHours`| **Yes** | **Keep** | Co-curricular involvement time recorded during the term. |
| `ClassParticipation` | **Yes** | **Keep** | Instructor assessment of student engagement prior to finals. |
| `PreviousBacklogs` | **Yes** | **Keep** | Historical academic transcript record of uncleared past courses. |
| `PostExamConfidence` | **NO (LEAKAGE)**| **DROP** | **CRITICAL LEAKAGE HAZARD:** Self-reported by students *after* writing the final exam. In any real-world prospective early-warning or student intervention system, post-exam confidence cannot exist before the exam. Retaining it would constitute temporal target leakage. |

#### Empirical Leakage Impact:
- Model trained **with** `PostExamConfidence`: CV RMSE = $6.592 \pm 0.490$, $R^2 = 0.775$.
- Model trained **without** `PostExamConfidence` (Honest Pre-Exam Pipeline): CV RMSE = $6.675 \pm 0.494$, $R^2 = 0.764$.
- Dropping this feature slightly reduces apparent score but guarantees genuine out-of-sample validity in production.

---

## 2. Clean Scikit-Learn Pipeline Architecture

All feature preprocessing, sentinel handling, imputation, scaling, and estimation are encapsulated within an atomic `sklearn.pipeline.Pipeline`:

```text
Raw Input DataFrame
       │
       ▼
[PreExamFeatureExtractor]   --> Extracts 8 justified pre-exam features (silently drops ID & PostExamConfidence)
       │
       ▼
[SentinelAndBoundsSanitizer]--> Replaces negative sentinels with NaN; winsorizes physical upper bounds
       │
       ▼
[SimpleImputer(median)]     --> Robust missing value imputation fitted strictly on training folds
       │
       ▼
[StandardScaler]            --> Mean centering and unit-variance normalization
       │
       ▼
[GradientBoostingRegressor] --> Final tuned ensemble estimator
```

**Correctness & Hygiene Guarantees:**
- Preprocessing parameters (medians, means, scales) are fitted **only** on the training folds inside each cross-validation split.
- Zero test data snooping or data contamination.
- Complete determinism via fixed random seeds (`random_state=42`).

---

## 3. Model Comparison & Cross-Validation Results

### 3.1 Validation Setup
- **Cross-Validation Strategy:** 5-Fold K-Fold Cross-Validation (`shuffle=True`, `random_state=42`).
- **Sample Size:** 998 clean training observations across 8 pre-exam features.
- **Reported Statistics:** Mean $\pm$ Standard Deviation across validation folds.

### 3.2 5-Fold Cross-Validation Comparison Table

| Approach | Model Family | CV RMSE (Mean $\pm$ Std) | CV MAE (Mean $\pm$ Std) | CV $R^2$ (Mean $\pm$ Std) |
| :--- | :--- | :---: | :---: | :---: |
| **1. Mean Baseline** | Null Model (`DummyRegressor`) | $13.989 \pm 1.194$ | $11.126 \pm 0.937$ | $-0.010 \pm 0.010$ |
| **2. Ridge Regression** | L2 Regularized Linear Model | $7.945 \pm 0.621$ | $5.960 \pm 0.352$ | $0.671 \pm 0.044$ |
| **3. Random Forest** | Bagging Tree Ensemble (100 trees) | $7.429 \pm 0.806$ | $5.714 \pm 0.550$ | $0.713 \pm 0.048$ |
| **4. Hist Gradient Boosting**| Binning Tree Ensemble | $7.168 \pm 0.739$ | $5.563 \pm 0.504$ | $0.734 \pm 0.025$ |
| **5. Gradient Boosting** | Tuned Sequential Boosting Ensemble | **$6.675 \pm 0.494$** | **$5.276 \pm 0.376$** | **$0.764 \pm 0.021$** |

### 3.3 Hyperparameter Tuning Budget & Decisions
- **Stated Compute Budget:** 25 iterations of `RandomizedSearchCV` across 5 folds = **125 distinct model fits**.
- **Search Space:**
  - `n_estimators`: `[80, 100, 120, 150, 180]`
  - `learning_rate`: `[0.03, 0.05, 0.08, 0.1, 0.12]`
  - `max_depth`: `[2, 3, 4]`
  - `subsample`: `[0.80, 0.85, 0.90, 0.95, 1.00]`
  - `min_samples_leaf`: `[2, 4, 6, 8]`
- **Optimal Hyperparameters:** `{'n_estimators': 150, 'learning_rate': 0.12, 'max_depth': 2, 'subsample': 0.95, 'min_samples_leaf': 6}`.
- **Outcome:** Reduced CV RMSE from $6.748$ to **$6.675$**, achieving lower variance ($\pm 0.494$) and superior generalization.

---

## 4. Error Analysis & Residual Diagnostics

Out-of-fold predictions ($N=998$) were generated across the 5 cross-validation folds to analyze residual behaviors: $\text{Residual} = y_{\text{true}} - \hat{y}_{\text{pred}}$.

```
Saved Diagnostic Plots:
- artifacts/residual_plot.png (Out-of-fold Actual vs Predicted, Residuals vs Predicted, Residual Distribution)
- artifacts/feature_importance.png (Permutation Feature Importance)
```

### 4.1 Subgroup / Segment Error Breakdown

#### Segment 1: Students with High Backlogs (Failing Segment)
| Backlog Cohort | Student Count | Mean Absolute Error (MAE) | Error Std | Mean Residual ($y - \hat{y}$) |
| :--- | :---: | :---: | :---: | :---: |
| 0 Backlogs | 468 | 4.75 | 3.78 | +0.23 |
| 1 Backlog | 299 | 5.43 | 4.28 | +0.18 |
| 2 Backlogs | 135 | 5.56 | 4.27 | -0.14 |
| **3+ Backlogs** | **88** | **6.50** | **5.51** | **-1.29** |

**Diagnosis & Why:**
- Students with 3 or more historical backlogs exhibit significantly elevated prediction error (MAE = $6.50$ vs $4.75$, error standard deviation = $5.51$).
- The negative mean residual ($-1.29$) confirms that the model **systematically overpredicts** performance for chronic backlog students.
- *Root Cause:* High backlogs carry compounded psychological distress, exam anxiety, and time-management deficits across multiple concurrent makeup exams that cannot be captured purely by basic study hour metrics.

#### Segment 2: Extreme Score Tails (Low Performers vs Top Scorers)
| Score Tier | Student Count | Mean Absolute Error (MAE) | Mean Residual ($y - \hat{y}$) | Primary Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **Low (< 60 pts)** | 157 | **7.50** | **-5.80** | Systematic **Overprediction** |
| Average (60–75 pts) | 377 | 4.57 | -0.89 | Well-calibrated |
| Good (75–90 pts) | 334 | 4.66 | +1.74 | Well-calibrated |
| **Top Tier (> 90 pts)** | 130 | **5.71** | **+5.29** | Systematic **Underprediction** |

**Diagnosis & Why:**
- Typical tree ensemble regression suffers from **shrinkage toward the sample mean** (73.7 pts).
- The model compresses the extremes, underestimating true perfectionists and overestimating failing students.

### 4.2 Permutation Feature Importance Ranking
Permutation importance measured by out-of-fold RMSE degradation ($N=15$ repeats):
1. **`PreviousExamScore`:** $+4.674 \pm 0.154$ (Dominant historical anchor)
2. **`PreviousBacklogs`:** $+3.388 \pm 0.120$ (Primary risk indicator)
3. **`SleepHours`:** $+2.682 \pm 0.113$ (Rest and cognitive readiness)
4. **`StudyHours`:** $+1.748 \pm 0.117$ (Active preparation)
5. **`ClassParticipation`:** $+1.433 \pm 0.070$ (Classroom engagement)
6. **`AssignmentsCompleted`:** $+1.147 \pm 0.088$ (Coursework discipline)
7. **`AttendancePercentage`:** $+0.879 \pm 0.052$ (Presence baseline)
8. **`ExtracurricularHours`:** $+0.157 \pm 0.023$ (Marginal direct effect)

---

## 5. Model Card (~Half Page)

### 5.1 Model Details & Architecture
- **Model Name:** Student Academic Performance Predictor (SAP-GB-v1.0).
- **Model Type:** Scikit-Learn Pipeline combining Median Imputation, Standard Scaling, and Tuned Gradient Boosted Regression Trees.
- **Release Date:** October 2026.
- **License / Ownership:** GDG NMIT Recruitment 5.0 Technical Challenge.

### 5.2 Intended Use
- **Primary Purpose:** Formative academic advising and early identification of undergraduate students at risk of underperforming on upcoming final exams.
- **Operational Window:** Administered 2 to 4 weeks prior to final examinations using verified semester coursework and attendance records.
- **Intended Users:** Academic advisors, teaching assistants, and university student mentorship cells.

### 5.3 Out-of-Scope & Prohibited Uses
- **Prohibited Use 1 (Automated Grading):** The model must **never** be used as a substitute for grading or determining actual transcript credit.
- **Prohibited Use 2 (Disciplinary Actions):** The model must **never** be used to deny exam admittance, revoke scholarships, or penalize students.
- **Prohibited Use 3 (Admissions Screening):** The model must **never** be applied to external applicants or non-collegiate populations.

### 5.4 Limitations & Data Biases
- **Cohort Bias:** Trained on an institutional sample of 1,000 collegiate students; may not generalize to different grading curricula, remote learning formats, or non-engineering disciplines.
- **Unmeasured Factors:** Model does not incorporate personal emergencies, illness, financial distress, or specific subject difficulty variations.
- **Tail Attenuation:** Tends to overestimate failing students by $\approx 5.8$ marks and underestimate top scorers by $\approx 5.3$ marks due to regression to the mean.

### 5.5 Failure Modes & Mitigations
- **Failure Mode (Extreme Sentinel Outliers):** Students with corrupt records (e.g. 99 study hours or -1 attendance).  
  *Mitigation:* Handled via upstream `SentinelAndBoundsSanitizer` clipping and median imputation.
- **Failure Mode (Overconfidence for At-Risk Students):** Chronic backlog students receive overly optimistic predictions.  
  *Mitigation:* The Streamlit UI incorporates explicit 95% Prediction Intervals ($\pm 13.1$ marks) and early warning alerts whenever predicted scores fall below 60.

---

## 6. How to Run & Verify

### 6.1 Requirements Installation
```bash
pip install -r requirements.txt
```

### 6.2 Train & Cross-Validate All Models
```bash
python -m src.train
```
*Outputs: 5-model CV summary table, tuning progress, `artifacts/model_pipeline.joblib`, and `artifacts/metrics_summary.json`.*

### 6.3 Run Error Analysis & Diagnostics
```bash
python -m src.evaluate
python -m src.explain
```
*Outputs: Segment error breakdown, `artifacts/residual_plot.png`, and `artifacts/feature_importance.png`.*

### 6.4 Batch Prediction CLI (`predict.py`)
```bash
python predict.py --input data/student_performance_test.csv --output submission.csv
```
*Output Format:* Verified strictly matching `ID,FinalExamScore`:
```csv
ID,FinalExamScore
101001,89.71
101002,84.49
101003,64.71
101004,71.19
101005,60.97
```

### 6.5 Interactive Streamlit Demo
```bash
streamlit run app.py
```
*Features: Interactive single-student score prediction with 95% uncertainty intervals, batch CSV file scoring with one-click export, and visual Model Card tabs.*
