from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud_detection import save_artifacts, train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the credit-card fraud detection soft-voting ensemble."
    )
    parser.add_argument("--data-path", type=str, required=True,
                        help="CSV containing V1-V28, Amount, and Class columns.")
    parser.add_argument("--model-path", type=str, default="ensemble_model.joblib")
    parser.add_argument("--scaler-path", type=str, default="scaler.joblib")
    parser.add_argument("--threshold-path", type=str, default="threshold.joblib")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.data_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = pd.read_csv(dataset_path)
    pipeline = train_pipeline(data, random_state=args.random_state)
    save_artifacts(
        pipeline["model"],
        pipeline["scaler"],
        args.model_path,
        args.scaler_path,
        threshold=pipeline["threshold"],
        threshold_path=args.threshold_path,
    )

    metrics = pipeline["metrics"]
    print("Training complete.")
    print(f"Saved threshold: {pipeline['threshold']:.4f}")
    print(f"Test precision: {metrics['precision']:.4f}")
    print(f"Test recall: {metrics['recall']:.4f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Test average precision / AUPRC: {metrics['auprc']:.4f}")
    print(f"Test MCC: {metrics['mcc']:.4f}")
    print(f"Test confusion matrix: {metrics['confusion_matrix']}")
    print("Precision-recall curve arrays are available in pipeline['metrics']['pr_curve'].")


if __name__ == "__main__":
    main()
