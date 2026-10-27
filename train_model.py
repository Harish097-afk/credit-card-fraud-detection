from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud_detection import save_artifacts, train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the credit-card fraud detection soft-voting ensemble.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to a CSV containing V1-V28, Amount, Time, and Class columns.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ensemble_model.joblib",
        help="Output path for the trained ensemble model.",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="scaler.joblib",
        help="Output path for the fitted StandardScaler.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible training.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.data_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = pd.read_csv(dataset_path)
    pipeline = train_pipeline(data, random_state=args.random_state)
    save_artifacts(pipeline["model"], pipeline["scaler"], args.model_path, args.scaler_path)

    metrics = pipeline["metrics"]
    print("Training complete.")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"AUPRC: {metrics['auprc']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
