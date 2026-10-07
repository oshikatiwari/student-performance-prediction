"""
Clean scikit-learn Pipeline and Preprocessing Architecture.
Strictly ensures:
- Column selection: only pre-exam justified features are processed.
- Leakage-proof: all transformers fitted only on training folds inside CV.
- Sentinel value cleanup (-1 converted to NaN) and physical domain bounding.
- Median imputation and standard scaling.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# The 8 justified, non-leaked, pre-exam predictive features
PRE_EXAM_FEATURES: List[str] = [
    "StudyHours",
    "AttendancePercentage",
    "PreviousExamScore",
    "AssignmentsCompleted",
    "SleepHours",
    "ExtracurricularHours",
    "ClassParticipation",
    "PreviousBacklogs"
]

# Physical plausible bounds for features
FEATURE_BOUNDS = {
    "StudyHours": (0.0, 24.0),
    "AttendancePercentage": (0.0, 100.0),
    "PreviousExamScore": (0.0, 100.0),
    "AssignmentsCompleted": (0.0, 100.0),
    "SleepHours": (0.0, 24.0),
    "ExtracurricularHours": (0.0, 24.0),
    "ClassParticipation": (0.0, 10.0),
    "PreviousBacklogs": (0.0, 15.0)
}


class PreExamFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Selects only pre-exam justified features from raw input DataFrame,
    silently dropping identifiers (e.g. ID) and leaked columns (e.g. PostExamConfidence).
    """
    def __init__(self, feature_names: List[str] = PRE_EXAM_FEATURES):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X
        # Select features if present; missing columns filled with NaN
        out = pd.DataFrame(index=X_df.index)
        for col in self.feature_names:
            if col in X_df.columns:
                out[col] = pd.to_numeric(X_df[col], errors="coerce")
            else:
                out[col] = np.nan
        return out


class SentinelAndBoundsSanitizer(BaseEstimator, TransformerMixin):
    """
    Sanitizes negative sentinel values (e.g., -1 for missing) by mapping them to NaN,
    and winsorizes/clips out-of-domain entries (e.g. StudyHours > 24, Attendance > 100)
    to realistic upper physical limits.
    """
    def __init__(self, bounds: dict = FEATURE_BOUNDS):
        self.bounds = bounds

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=PRE_EXAM_FEATURES)

        for col, (low, high) in self.bounds.items():
            if col in X_df.columns:
                # Sentinel negative values treated as unobserved
                mask_negative = X_df[col] < low
                X_df.loc[mask_negative, col] = np.nan
                # Upper bound clipping
                X_df[col] = X_df[col].clip(upper=high)

        return X_df.values


def build_preprocessor() -> Pipeline:
    """Builds the end-to-end feature preprocessing sub-pipeline."""
    return Pipeline([
        ("feature_selection", PreExamFeatureExtractor(PRE_EXAM_FEATURES)),
        ("sanitizer", SentinelAndBoundsSanitizer(FEATURE_BOUNDS)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


def build_pipeline(model=None) -> Pipeline:
    """Constructs the complete sklearn Pipeline combining preprocessing and estimator."""
    steps = [
        ("preprocessor", build_preprocessor())
    ]
    if model is not None:
        steps.append(("model", model))
    return Pipeline(steps)
