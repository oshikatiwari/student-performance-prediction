#!/usr/bin/env python3
"""
Packaged Inference CLI for Student Performance Prediction.
Usage:
    python predict.py --input data/student_performance_test.csv --output submission.csv
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batch inference for Student Performance Prediction pipeline."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input CSV file containing student feature records."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="submission.csv",
        help="Path to output CSV file for predictions (default: submission.csv)."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="artifacts/model_pipeline.joblib",
        help="Path to serialized joblib pipeline artifact (default: artifacts/model_pipeline.joblib)."
    )
    return parser.parse_args()


def run_inference(input_path: str, output_path: str, model_path: str = "artifacts/model_pipeline.joblib"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at: {model_path}. Please run 'python -m src.train' first.")

    # 1. Load input dataset
    df_input = pd.read_csv(input_path)
    n_rows = len(df_input)

    # 2. Extract or generate ID column
    if "ID" in df_input.columns:
        ids = df_input["ID"].astype(int)
    else:
        ids = pd.Series(range(100001, 100001 + n_rows), name="ID")

    # 3. Load trained model pipeline
    pipeline = joblib.load(model_path)

    # 4. Generate predictions
    raw_predictions = pipeline.predict(df_input)

    # 5. Post-process: clip to valid exam score domain [0.0, 100.0] and round to 2 decimals
    clipped_predictions = np.clip(raw_predictions, 0.0, 100.0)
    final_scores = np.round(clipped_predictions, 2)

    # 6. Format submission DataFrame strictly as ID,FinalExamScore
    df_output = pd.DataFrame({
        "ID": ids,
        "FinalExamScore": final_scores
    })

    # 7. Ensure output directory exists and write CSV
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_output.to_csv(output_path, index=False)
    print(f"Successfully generated predictions for {len(df_output)} records.")
    print(f"Output saved to: {output_path}")
    print("\nPreview of predictions:")
    print(df_output.head(5).to_string(index=False))


def main():
    args = parse_args()
    try:
        run_inference(args.input, args.output, args.model)
    except Exception as e:
        print(f"Inference Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
