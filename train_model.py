from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--report-path", type=str, default="evaluation_report.json",
                        help="Where to save dataset summary and held-out test metrics.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.data_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = pd.read_csv(dataset_path)
    if "Class" not in data.columns:
        raise ValueError("Dataset must contain a 'Class' target column.")

    class_counts = data["Class"].value_counts(dropna=False)
    fraud_count = int((data["Class"] == 1).sum())
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
    tn, fp, fn, tp = metrics["confusion_matrix"][0][0], metrics["confusion_matrix"][0][1], metrics["confusion_matrix"][1][0], metrics["confusion_matrix"][1][1]
    report = {
        "dataset": {
            "path": str(dataset_path),
            "rows": int(len(data)),
            "fraud_count": fraud_count,
            "legitimate_count": int((data["Class"] == 0).sum()),
            "fraud_rate": float(fraud_count / len(data)) if len(data) else 0.0,
            "class_values": {str(label): int(count) for label, count in class_counts.items()},
        },
        "threshold": float(pipeline["threshold"]),
        "test_metrics": {
            key: value for key, value in metrics.items() if key != "pr_curve"
        },
        "confusion_matrix_counts": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "note": "Threshold selected on validation data; metrics reported on the held-out test split.",
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Training and held-out evaluation complete.")
    print(f"Dataset rows: {len(data):,}")
    print(f"Fraud transactions: {fraud_count:,} ({fraud_count / len(data):.4%})")
    print(f"Saved threshold: {pipeline['threshold']:.4f}")
    print(f"Test precision: {metrics['precision']:.4f}")
    print(f"Test recall: {metrics['recall']:.4f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Test average precision / AUPRC: {metrics['auprc']:.4f}")
    print(f"Test MCC: {metrics['mcc']:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")
    print(f"Saved evaluation report: {report_path}")


if __name__ == "__main__":
    main()
