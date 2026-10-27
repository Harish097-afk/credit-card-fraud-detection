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
    precision_recall_curve,
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
DEFAULT_THRESHOLD = 0.5


def _validate_required_columns(frame: pd.DataFrame, *, for_inference: bool = False) -> None:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        missing_label = "for inference" if for_inference else "for training"
        raise ValueError(f"Missing required feature columns {missing_label}: {missing}")

    if AMOUNT_COLUMN not in frame.columns and "scaled_amount" not in frame.columns:
        raise ValueError("The input data must contain an 'Amount' column or a 'scaled_amount' column.")


def prepare_training_frame(
    frame: pd.DataFrame, scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Drop non-features and standardize raw Amount using a train-fitted scaler."""
    working = frame.copy()
    working = working.drop(columns=[c for c in (TARGET_COLUMN, "Time") if c in working.columns])
    _validate_required_columns(working)

    if AMOUNT_COLUMN not in working.columns:
        raise ValueError("Training data must contain raw 'Amount'; scaled_amount is inference-only.")

    working[AMOUNT_COLUMN] = pd.to_numeric(working[AMOUNT_COLUMN], errors="raise")
    if scaler is None:
        scaler = StandardScaler()
        working[AMOUNT_COLUMN] = scaler.fit_transform(working[[AMOUNT_COLUMN]]).ravel()
    else:
        working[AMOUNT_COLUMN] = scaler.transform(working[[AMOUNT_COLUMN]]).ravel()

    return working[FEATURE_COLUMNS + [AMOUNT_COLUMN]], scaler


def prepare_inference_frame(frame: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Prepare inference features without scaling an already-standardized amount twice."""
    working = frame.copy()
    working = working.drop(columns=[c for c in (TARGET_COLUMN, "Time") if c in working.columns])
    _validate_required_columns(working, for_inference=True)

    if AMOUNT_COLUMN in working.columns:
        raw_amount = pd.to_numeric(working[AMOUNT_COLUMN], errors="raise")
        working[AMOUNT_COLUMN] = scaler.transform(raw_amount.to_frame()).ravel()
    else:
        working[AMOUNT_COLUMN] = pd.to_numeric(working["scaled_amount"], errors="raise")

    prepared = working[FEATURE_COLUMNS + [AMOUNT_COLUMN]]
    if not np.isfinite(prepared.to_numpy(dtype=float)).all():
        raise ValueError("Inference features contain NaN or infinite values.")
    return prepared


def make_soft_voting_model() -> VotingClassifier:
    """Create the project's soft-voting ensemble."""
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
    )
    lgbm_model = LGBMClassifier(
        n_estimators=100, learning_rate=0.05, num_leaves=31,
        objective="binary", random_state=42, n_jobs=-1,
    )
    xgb_model = XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=6,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    return VotingClassifier(
        estimators=[("rf", rf_model), ("lgbm", lgbm_model), ("xgb", xgb_model)],
        voting="soft",
    )


def _threshold_metrics(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> Dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, probabilities)
    average_precision = float(average_precision_score(y_true, probabilities))
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": average_precision,
        "auprc": average_precision,
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "pr_curve": {
            "precision": precision.astype(float).tolist(),
            "recall": recall.astype(float).tolist(),
            "thresholds": pr_thresholds.astype(float).tolist(),
        },
    }


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Select the threshold maximizing F1 on validation data."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return DEFAULT_THRESHOLD
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def evaluate_model(
    model: VotingClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Evaluate predictions at a chosen threshold."""
    probabilities = model.predict_proba(X_test)[:, 1]
    return _threshold_metrics(y_test, probabilities, threshold)


def train_pipeline(frame: pd.DataFrame, random_state: int = 42) -> Dict[str, Any]:
    """Train, tune threshold on validation data, and evaluate on held-out test data."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"The dataset must contain the target column '{TARGET_COLUMN}'.")

    working = frame.copy()
    if "Time" in working.columns:
        working = working.drop(columns=["Time"])
    X = working.drop(columns=[TARGET_COLUMN])
    y = working[TARGET_COLUMN].astype(int)
    if y.nunique() != 2:
        raise ValueError("The target column must contain both classes.")

    X_train_val, X_test_raw, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25,
        stratify=y_train_val, random_state=random_state
    )

    X_train, scaler = prepare_training_frame(X_train_raw)
    X_val, _ = prepare_training_frame(X_val_raw, scaler=scaler)
    X_test, _ = prepare_training_frame(X_test_raw, scaler=scaler)

    # SMOTE is applied only to the training partition to avoid leakage.
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    ensemble_model = make_soft_voting_model()
    ensemble_model.fit(X_train_resampled, y_train_resampled)

    validation_probabilities = ensemble_model.predict_proba(X_val)[:, 1]
    best_threshold = select_threshold(y_val, validation_probabilities)
    validation_metrics = _threshold_metrics(y_val, validation_probabilities, best_threshold)
    test_metrics = evaluate_model(ensemble_model, X_test, y_test, threshold=best_threshold)

    return {
        "model": ensemble_model,
        "scaler": scaler,
        "threshold": best_threshold,
        "validation_metrics": validation_metrics,
        "metrics": test_metrics,
        "X_test": X_test,
        "y_test": y_test,
    }


def save_artifacts(
    model: VotingClassifier,
    scaler: StandardScaler,
    model_path: str = "ensemble_model.joblib",
    scaler_path: str = "scaler.joblib",
    threshold: Optional[float] = None,
    threshold_path: str = "threshold.joblib",
) -> None:
    """Save model/scaler and, when supplied, the validation-selected threshold."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    if threshold is not None:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        joblib.dump(float(threshold), threshold_path)


def load_threshold(threshold_path: str = "threshold.joblib") -> float:
    """Load the saved decision threshold, falling back for older model artifacts."""
    path = Path(threshold_path)
    if not path.exists():
        return DEFAULT_THRESHOLD
    threshold = float(joblib.load(path))
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid saved threshold in {path}; expected a value from 0 to 1.")
    return threshold


def load_artifacts(
    model_path: str = "ensemble_model.joblib", scaler_path: str = "scaler.joblib"
) -> Tuple[Optional[Any], Optional[StandardScaler]]:
    model_file, scaler_file = Path(model_path), Path(scaler_path)
    if not model_file.exists() or not scaler_file.exists():
        return None, None
    return joblib.load(model_file), joblib.load(scaler_file)


__all__ = [
    "AMOUNT_COLUMN", "DEFAULT_THRESHOLD", "FEATURE_COLUMNS", "TARGET_COLUMN",
    "evaluate_model", "load_artifacts", "load_threshold", "make_soft_voting_model",
    "prepare_inference_frame", "prepare_training_frame", "save_artifacts",
    "select_threshold", "train_pipeline",
]


if __name__ == "__main__":
    print("Use train_model.py to train and export the ensemble model artifacts.")
