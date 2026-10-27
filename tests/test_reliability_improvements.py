import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fraud_detection import (
    AMOUNT_COLUMN,
    FEATURE_COLUMNS,
    prepare_inference_frame,
    select_threshold,
)


def _features(amount_column, values):
    data = {feature: [0.1, -0.2] for feature in FEATURE_COLUMNS}
    data[amount_column] = values
    return pd.DataFrame(data)


def test_raw_amount_is_scaled_once_with_training_scaler():
    scaler = StandardScaler().fit(pd.DataFrame({AMOUNT_COLUMN: [0.0, 10.0, 20.0]}))
    raw = _features(AMOUNT_COLUMN, [10.0, 20.0])

    prepared = prepare_inference_frame(raw, scaler)

    np.testing.assert_allclose(
        prepared[AMOUNT_COLUMN].to_numpy(),
        scaler.transform(pd.DataFrame({AMOUNT_COLUMN: [10.0, 20.0]})).ravel(),
    )


def test_scaled_amount_is_not_scaled_a_second_time():
    scaler = StandardScaler().fit(pd.DataFrame({AMOUNT_COLUMN: [100.0, 200.0, 300.0]}))
    already_scaled = _features("scaled_amount", [-1.25, 0.75])

    prepared = prepare_inference_frame(already_scaled, scaler)

    np.testing.assert_allclose(prepared[AMOUNT_COLUMN].to_numpy(), [-1.25, 0.75])
    assert prepared.columns.tolist() == FEATURE_COLUMNS + [AMOUNT_COLUMN]


def test_threshold_selection_uses_validation_probabilities():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.70, 0.90])

    threshold = select_threshold(y_true, probabilities)

    assert 0.20 <= threshold <= 0.90
    predictions = (probabilities >= threshold).astype(int)
    assert (predictions == y_true.to_numpy()).all()


def test_inference_rejects_non_numeric_amount():
    scaler = StandardScaler().fit(pd.DataFrame({AMOUNT_COLUMN: [1.0, 2.0]}))
    data = _features(AMOUNT_COLUMN, ["not-a-number", "2.0"])

    try:
        prepare_inference_frame(data, scaler)
    except (ValueError, TypeError):
        return
    raise AssertionError("Expected invalid Amount values to be rejected")
