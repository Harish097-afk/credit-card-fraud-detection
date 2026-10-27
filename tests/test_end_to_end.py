import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection import (
    AMOUNT_COLUMN,
    FEATURE_COLUMNS,
    load_artifacts,
    load_threshold,
    prepare_inference_frame,
    save_artifacts,
    train_pipeline,
)


def _synthetic_dataset(rows=100, fraud_rows=30, seed=19):
    rng = np.random.default_rng(seed)
    labels = np.array([1] * fraud_rows + [0] * (rows - fraud_rows))
    rng.shuffle(labels)
    features = rng.normal(0, 1, size=(rows, len(FEATURE_COLUMNS)))
    # Add a learnable signal without relying on any external dataset.
    features[:, 0] += labels * 2.0
    features[:, 1] -= labels * 1.5
    data = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    data[AMOUNT_COLUMN] = rng.uniform(1, 500, size=rows)
    data["Class"] = labels
    return data


def test_end_to_end_training_single_batch_and_csv_export(tmp_path):
    data = _synthetic_dataset()

    pipeline = train_pipeline(data, random_state=42)
    assert 0.0 <= pipeline["threshold"] <= 1.0
    assert 0.0 <= pipeline["metrics"]["precision"] <= 1.0
    assert 0.0 <= pipeline["metrics"]["recall"] <= 1.0

    model_path = tmp_path / "ensemble_model.joblib"
    scaler_path = tmp_path / "scaler.joblib"
    threshold_path = tmp_path / "threshold.joblib"
    save_artifacts(
        pipeline["model"],
        pipeline["scaler"],
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        threshold=pipeline["threshold"],
        threshold_path=str(threshold_path),
    )

    model, scaler = load_artifacts(str(model_path), str(scaler_path))
    assert model is not None
    assert scaler is not None
    threshold = load_threshold(str(threshold_path))
    assert threshold == pipeline["threshold"]

    input_rows = data.drop(columns=["Class", "Time"], errors="ignore").iloc[:5].copy()
    prepared_batch = prepare_inference_frame(input_rows, scaler)
    batch_probabilities = model.predict_proba(prepared_batch)[:, 1]
    batch_predictions = (batch_probabilities >= threshold).astype(int)

    assert len(batch_probabilities) == 5
    assert np.isfinite(batch_probabilities).all()
    assert set(batch_predictions).issubset({0, 1})

    prepared_single = prepare_inference_frame(input_rows.iloc[[0]], scaler)
    single_probability = float(model.predict_proba(prepared_single)[0, 1])
    single_prediction = int(single_probability >= threshold)
    assert 0.0 <= single_probability <= 1.0
    assert single_prediction in (0, 1)

    output = input_rows.copy()
    output["Fraud_Prediction"] = batch_predictions
    output["Fraud_Probability"] = batch_probabilities
    output["Decision_Threshold"] = threshold
    output["Fraud_Label"] = [
        "Fraud" if value else "Legitimate" for value in batch_predictions
    ]
    csv_path = tmp_path / "fraud_detection_results.csv"
    output.to_csv(csv_path, index=False)
    restored = pd.read_csv(csv_path)
    assert len(restored) == len(input_rows)
    assert {"Fraud_Prediction", "Fraud_Probability", "Decision_Threshold", "Fraud_Label"}.issubset(
        restored.columns
    )
    assert restored["Fraud_Probability"].between(0, 1).all()
