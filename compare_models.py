from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud_detection import FEATURE_COLUMNS, TARGET_COLUMN, select_threshold


def evaluate(name, model, X_val, y_val, X_test, y_test):
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val, val_prob)
    test_prob = model.predict_proba(X_test)[:, 1]
    pred = (test_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "threshold": float(threshold),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
        "auprc": float(average_precision_score(y_test, test_prob)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare fraud models on identical stratified train/validation/test splits."
    )
    parser.add_argument("--data-path", required=True, help="Path to creditcard.csv")
    parser.add_argument("--report-path", default="model_comparison.json")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    required = FEATURE_COLUMNS + ["Amount", TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    data = data[required].copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains missing or infinite values.")
    if set(data[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Class must contain both labels 0 and 1.")

    X = data[FEATURE_COLUMNS + ["Amount"]]
    y = data[TARGET_COLUMN].astype(int)
    X_trainval, X_test_raw, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=args.random_state
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval,
        random_state=args.random_state
    )

    scaler = StandardScaler()
    X_train = X_train_raw.copy()
    X_val = X_val_raw.copy()
    X_test = X_test_raw.copy()
    X_train["Amount"] = scaler.fit_transform(X_train_raw[["Amount"]]).ravel()
    X_val["Amount"] = scaler.transform(X_val_raw[["Amount"]]).ravel()
    X_test["Amount"] = scaler.transform(X_test_raw[["Amount"]]).ravel()

    # Resample only the training partition. Validation/test retain the real class balance.
    X_resampled, y_resampled = SMOTE(random_state=args.random_state).fit_resample(
        X_train, y_train
    )

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=12, n_jobs=-1, random_state=args.random_state
    )
    lgbm = LGBMClassifier(
        n_estimators=100, learning_rate=0.05, num_leaves=31,
        objective="binary", random_state=args.random_state, n_jobs=-1,
        verbosity=-1,
    )
    xgb = XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=6,
        objective="binary:logistic", eval_metric="logloss",
        random_state=args.random_state, n_jobs=-1,
    )
    models = [
        ("Logistic Regression", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=args.random_state
        )),
        ("Random Forest", rf),
        ("LightGBM", lgbm),
        ("XGBoost", xgb),
        ("Soft Voting Ensemble", VotingClassifier(
            estimators=[("rf", rf), ("lgbm", lgbm), ("xgb", xgb)], voting="soft"
        )),
    ]

    results = []
    for name, model in models:
        print(f"Training {name}...", flush=True)
        model.fit(X_resampled, y_resampled)
        results.append(evaluate(name, model, X_val, y_val, X_test, y_test))

    results.sort(key=lambda row: row["auprc"], reverse=True)
    report = {
        "dataset_rows": int(len(data)),
        "fraud_count": int((y == 1).sum()),
        "legitimate_count": int((y == 0).sum()),
        "split": {"train": int(len(y_train)), "validation": int(len(y_val)), "test": int(len(y_test))},
        "threshold_selection": "F1 maximization on validation data only",
        "test_metrics": results,
        "caution": "The test split is stratified random, not chronological. Compare with a time-ordered evaluation before making deployment claims.",
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nHeld-out test comparison (sorted by AUPRC):")
    print(f"{'Model':24} {'Prec.':>8} {'Recall':>8} {'AUPRC':>8} {'ROC-AUC':>8} {'FP':>7} {'FN':>7}")
    for row in results:
        print(f"{row['model']:24} {row['precision']:8.4f} {row['recall']:8.4f} "
              f"{row['auprc']:8.4f} {row['roc_auc']:8.4f} "
              f"{row['false_positives']:7d} {row['false_negatives']:7d}")
    print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()
