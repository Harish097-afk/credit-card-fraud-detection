from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection import (
    AMOUNT_COLUMN,
    FEATURE_COLUMNS,
    make_soft_voting_model,
    prepare_inference_frame,
    train_pipeline,
)


def _make_synthetic_dataset(n_legit: int = 180, n_fraud: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []

    for _ in range(n_legit):
        row = {f"V{i}": float(rng.normal(0, 1)) for i in range(1, 29)}
        row[AMOUNT_COLUMN] = float(rng.uniform(1.0, 500.0))
        row["Time"] = float(rng.uniform(0, 200000))
        row["Class"] = 0
        rows.append(row)

    for _ in range(n_fraud):
        row = {f"V{i}": float(rng.normal(1.5, 1.2)) for i in range(1, 29)}
        row[AMOUNT_COLUMN] = float(rng.uniform(50.0, 1000.0))
        row["Time"] = float(rng.uniform(0, 200000))
        row["Class"] = 1
        rows.append(row)

    return pd.DataFrame(rows)


def test_make_soft_voting_model_uses_soft_voting() -> None:
    model = make_soft_voting_model()
    assert model.voting == "soft"
    assert len(model.estimators) == 3


def test_prepare_inference_frame_keeps_expected_schema() -> None:
    data = pd.DataFrame({
        **{feature: [0.1, -0.2] for feature in FEATURE_COLUMNS},
        AMOUNT_COLUMN: [100.0, 50.0],
    })

    scaler = __import__("sklearn.preprocessing").preprocessing.StandardScaler()
    scaler.fit(data[[AMOUNT_COLUMN]])

    prepared = prepare_inference_frame(data, scaler)
    assert prepared.columns.tolist() == FEATURE_COLUMNS + [AMOUNT_COLUMN]
    assert prepared.shape == (2, 30)
    assert np.isfinite(prepared.to_numpy()).all()


def test_train_pipeline_runs_and_returns_metrics() -> None:
    dataset = _make_synthetic_dataset()
    result = train_pipeline(dataset, random_state=42)

    assert "model" in result
    assert "scaler" in result
    assert "metrics" in result
    metric_keys = {"precision", "recall", "roc_auc", "average_precision", "auprc", "mcc", "confusion_matrix"}
    assert metric_keys.issubset(result["metrics"].keys())
    assert result["model"].voting == "soft"
    assert len(result["model"].estimators) == 3
