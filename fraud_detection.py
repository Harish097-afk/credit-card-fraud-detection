from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)]
AMOUNT_COLUMN = "Amount"
TARGET_COLUMN = "Class"


def _validate_required_columns(frame: pd.DataFrame, *, for_inference: bool = False) -> None:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        missing_label = "for inference" if for_inference else "for training"
        raise ValueError(f"Missing required feature columns {missing_label}: {missing}")

    if AMOUNT_COLUMN not in frame.columns and "scaled_amount" not in frame.columns:
        raise ValueError("The input data must contain an 'Amount' column or a 'scaled_amount' column.")


def prepare_training_frame(frame: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """Prepare the training dataset by dropping unlabeled columns and scaling Amount."""
    working = frame.copy()
    if TARGET_COLUMN in working.columns:
        working = working.drop(columns=[TARGET_COLUMN])
    if "Time" in working.columns:
        working = working.drop(columns=["Time"])

    _validate_required_columns(working)

    if scaler is None:
        scaler = StandardScaler()
        working[AMOUNT_COLUMN] = scaler.fit_transform(working[[AMOUNT_COLUMN]]).ravel()
    else:
        working[AMOUNT_COLUMN] = scaler.transform(working[[AMOUNT_COLUMN]]).ravel()

    return working[FEATURE_COLUMNS + [AMOUNT_COLUMN]], scaler


def prepare_inference_frame(frame: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Return inference data with the exact schema used to train the ensemble."""
    working = frame.copy()
    if TARGET_COLUMN in working.columns:
        working = working.drop(columns=[TARGET_COLUMN])
    if "Time" in working.columns:
        working = working.drop(columns=["Time"])

    _validate_required_columns(working, for_inference=True)

    if AMOUNT_COLUMN not in working.columns:
        working[AMOUNT_COLUMN] = working["scaled_amount"]

    working[AMOUNT_COLUMN] = scaler.transform(working[[AMOUNT_COLUMN]]).ravel()
    return working[FEATURE_COLUMNS + [AMOUNT_COLUMN]]


def make_soft_voting_model() -> VotingClassifier:
    """Create the soft-voting ensemble defined in the project specification."""
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
    )

    lgbm_model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        objective="binary",
        random_state=42,
        n_jobs=-1,
    )

    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    return VotingClassifier(
        estimators=[
            ("rf", rf_model),
            ("lgbm", lgbm_model),
            ("xgb", xgb_model),
        ],
        voting="soft",
    )


def evaluate_model(model: VotingClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Compute the evaluation metrics requested for fraud detection benchmarking."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics: Dict[str, Any] = {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "auprc": float(average_precision_score(y_test, probabilities)),
        "mcc": float(matthews_corrcoef(y_test, predictions)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    return metrics


def train_pipeline(frame: pd.DataFrame, random_state: int = 42) -> Dict[str, Any]:
    """Train the full preprocessing + SMOTE + soft-voting ensemble pipeline."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"The dataset must contain the target column '{TARGET_COLUMN}'.")

    working = frame.copy()
    if "Time" in working.columns:
        working = working.drop(columns=["Time"])

    X = working.drop(columns=[TARGET_COLUMN])
    y = working[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        raise ValueError("The target column must contain both classes in train and test splits.")

    X_train, scaler = prepare_training_frame(X_train, scaler=None)
    X_test, _ = prepare_training_frame(X_test, scaler=scaler)

    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    ensemble_model = make_soft_voting_model()
    ensemble_model.fit(X_train_resampled, y_train_resampled)

    metrics = evaluate_model(ensemble_model, X_test, y_test)
    return {
        "model": ensemble_model,
        "scaler": scaler,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
    }


def save_artifacts(model: VotingClassifier, scaler: StandardScaler, model_path: str = "ensemble_model.joblib", scaler_path: str = "scaler.joblib") -> None:
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


def load_artifacts(model_path: str = "ensemble_model.joblib", scaler_path: str = "scaler.joblib") -> Tuple[Optional[Any], Optional[StandardScaler]]:
    model_file = Path(model_path)
    scaler_file = Path(scaler_path)

    if not model_file.exists() or not scaler_file.exists():
        return None, None

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    return model, scaler


__all__ = [
    "AMOUNT_COLUMN",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "evaluate_model",
    "load_artifacts",
    "make_soft_voting_model",
    "prepare_inference_frame",
    "prepare_training_frame",
    "save_artifacts",
    "train_pipeline",
]


if __name__ == "__main__":
    print("fraud_detection.py - reusable training and inference pipeline for the credit-card fraud detection project.")
    print("Use train_model.py to train and export the ensemble model artifacts.")
    
    