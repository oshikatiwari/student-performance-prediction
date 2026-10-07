"""
Data Audit and Pre-Exam Feature Justification Module.
Provides comprehensive audit of student_performance.csv:
- Shape, missingness, distribution stats
- Bad records and sentinel value identification
- Feature leakage check: determines pre-exam availability and keep/drop justification
"""

import os
import pandas as pd
import numpy as np


FEATURE_JUSTIFICATIONS = {
    "ID": {
        "pre_exam": False,
        "action": "Drop",
        "justification": "Arbitrary student identifier; contains no academic signal and would cause spurious correlation."
    },
    "StudyHours": {
        "pre_exam": True,
        "action": "Keep",
        "justification": "Recorded hours dedicated to self-study prior to the examination. Valid operational behavioral predictor."
    },
    "AttendancePercentage": {
        "pre_exam": True,
        "action": "Keep",
        "justification": "Cumulative semester classroom attendance up to exam eligibility threshold. Known pre-exam."
    },
    "PreviousExamScore": {
        "pre_exam": True,
        "action": "Keep",
        "justification": "Historical score from midterms or prerequisite exams prior to finals. Valid baseline signal."
    },
    "AssignmentsCompleted": {
        "pre_exam": True,
        "action": "Keep",
        "justification": "Percentage of regular coursework submissions completed throughout the term before finals."
    },
    "SleepHours": {
        "pre_exam": True,
        "action": "Keep",
        "justification": "Routine daily sleep duration reflecting physical wellbeing and readiness prior to exam day."
    },
    "ExtracurricularHours": {
        "pre_exam": True,
        "action": "Keep",
        "justification": "Weekly hours spent in sports, cultural, or community clubs prior to examination."
    },
    "ClassParticipation": {
        "pre_exam": True,
        "action": "Keep",
        "justification": "Continuous in-class engagement score evaluated by instructors throughout the term."
    },
    "PreviousBacklogs": {
        "pre_exam": True,
        "action": "Keep",
        "justification": "Official count of pending course backlogs from prior semesters on student record before the exam."
    },
    "PostExamConfidence": {
        "pre_exam": False,
        "action": "Drop",
        "justification": "CRITICAL TARGET LEAKAGE: Self-reported student confidence solicited *after* writing the final exam. "
                         "Cannot realistically be known before the exam in a real-world early-warning / intervention system. "
                         "Must be dropped to prevent temporal data leakage."
    },
    "FinalExamScore": {
        "pre_exam": False,
        "action": "Target",
        "justification": "Supervised continuous prediction target (0 - 100)."
    }
}


def run_data_audit(csv_path: str = "data/student_performance.csv") -> dict:
    """Performs deep data audit and returns a structured audit report dictionary."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    audit_report = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_counts": df.isnull().sum().to_dict(),
        "missing_percentages": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "target_stats": {},
        "bad_records": {},
        "feature_justifications": FEATURE_JUSTIFICATIONS
    }

    # Target statistics
    if "FinalExamScore" in df.columns:
        s = df["FinalExamScore"]
        audit_report["target_stats"] = {
            "mean": round(float(s.mean()), 2),
            "std": round(float(s.std()), 2),
            "median": round(float(s.median()), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "q25": round(float(s.quantile(0.25)), 2),
            "q75": round(float(s.quantile(0.75)), 2),
        }

    # Bad record checks
    # Corrupt target: score < 0 or > 100
    invalid_targets = df[(df["FinalExamScore"] < 0) | (df["FinalExamScore"] > 100)]
    audit_report["bad_records"]["invalid_target_rows"] = invalid_targets[["ID", "FinalExamScore"]].to_dict(orient="records")

    # Sentinel / impossible feature values
    neg_study = df[df["StudyHours"] < 0]["ID"].tolist()
    excess_study = df[df["StudyHours"] > 24]["ID"].tolist()
    excess_attendance = df[df["AttendancePercentage"] > 100]["ID"].tolist()
    neg_attendance = df[df["AttendancePercentage"] < 0]["ID"].tolist()
    excess_prev_score = df[df["PreviousExamScore"] > 100]["ID"].tolist()
    excess_assignments = df[df["AssignmentsCompleted"] > 100]["ID"].tolist()
    excess_sleep = df[df["SleepHours"] > 24]["ID"].tolist()
    neg_backlogs = df[df["PreviousBacklogs"] < 0]["ID"].tolist()

    audit_report["bad_records"]["anomalies"] = {
        "negative_study_hours": neg_study,
        "excess_study_hours_gt_24": excess_study,
        "attendance_gt_100": excess_attendance,
        "attendance_lt_0": neg_attendance,
        "prev_exam_score_gt_100": excess_prev_score,
        "assignments_gt_100": excess_assignments,
        "sleep_hours_gt_24": excess_sleep,
        "negative_backlogs": neg_backlogs
    }

    return audit_report


def print_audit_summary(audit_report: dict):
    print("=" * 70)
    print("           GDG NMIT - DATA AUDIT & LEAKAGE CHECK REPORT")
    print("=" * 70)
    print(f"Dataset Shape: {audit_report['shape'][0]} rows x {audit_report['shape'][1]} columns\n")
    
    print("--- Missing Values Summary ---")
    for col, count in audit_report["missing_counts"].items():
        pct = audit_report["missing_percentages"][col]
        print(f"  {col:22s}: {count:3d} missing ({pct:4.1f}%)")

    print("\n--- Target Distribution (FinalExamScore) ---")
    ts = audit_report["target_stats"]
    print(f"  Mean: {ts.get('mean')}, Std: {ts.get('std')}, Median: {ts.get('median')}")
    print(f"  Min: {ts.get('min')} (Anomalous!), Max: {ts.get('max')} (Anomalous!), IQR: [{ts.get('q25')} - {ts.get('q75')}]")

    print("\n--- Bad Records & Corrupt Values ---")
    bad_t = audit_report["bad_records"]["invalid_target_rows"]
    print(f"  Invalid Target Labels (Must be dropped from training): {len(bad_t)} records")
    for row in bad_t:
        print(f"    ID {row['ID']}: FinalExamScore = {row['FinalExamScore']}")

    print("\n  Out-of-Bounds & Sentinel Feature Anomalies:")
    for k, v in audit_report["bad_records"]["anomalies"].items():
        if v:
            print(f"    {k:28s}: {len(v)} occurrences (IDs: {v})")

    print("\n--- Feature Justification & Leakage Assessment ---")
    print(f"{'Feature':<22s} | {'Pre-Exam?':<10s} | {'Action':<6s} | {'Justification'}")
    print("-" * 70)
    for col, info in audit_report["feature_justifications"].items():
        pre = "Yes" if info["pre_exam"] else "No"
        print(f"{col:<22s} | {pre:<10s} | {info['action']:<6s} | {info['justification']}")
    print("=" * 70)


if __name__ == "__main__":
    report = run_data_audit("data/student_performance.csv")
    print_audit_summary(report)
