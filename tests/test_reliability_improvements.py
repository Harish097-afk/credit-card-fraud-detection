import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from fraud_detection import (
    AMOUNT_COLUMN,
    FEATURE_COLUMNS,
    DEFAULT_THRESHOLD,
    load_threshold,
    prepare_inference_frame,
    save_artifacts,
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

    with pytest.raises((ValueError, TypeError)):
        prepare_inference_frame(data, scaler)


def test_missing_threshold_artifact_uses_backward_compatible_default(tmp_path):
    assert load_threshold(str(tmp_path / "missing.joblib")) == DEFAULT_THRESHOLD


def test_saved_threshold_round_trips_and_is_validated(tmp_path):
    import joblib

    threshold_path = tmp_path / "threshold.joblib"
    joblib.dump(0.37, threshold_path)
    assert load_threshold(str(threshold_path)) == pytest.approx(0.37)

    joblib.dump(1.5, threshold_path)
    with pytest.raises(ValueError, match="Invalid saved threshold"):
        load_threshold(str(threshold_path))


def test_save_artifacts_persists_threshold(tmp_path):
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
    scaler = StandardScaler().fit([[0.0], [1.0]])
    threshold_path = tmp_path / "threshold.joblib"
    save_artifacts(
        model, scaler,
        model_path=str(tmp_path / "model.joblib"),
        scaler_path=str(tmp_path / "scaler.joblib"),
        threshold=0.42,
        threshold_path=str(threshold_path),
    )
    assert load_threshold(str(threshold_path)) == pytest.approx(0.42)


def _training_frame(rows=80, fraud_count=20):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        rng.normal(size=(rows, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )
    frame[AMOUNT_COLUMN] = rng.uniform(1, 500, size=rows)
    frame["Class"] = [1] * fraud_count + [0] * (rows - fraud_count)
    return frame


def test_training_rejects_missing_required_feature():
    from fraud_detection import _validate_training_data

    frame = _training_frame().drop(columns=["V7"])
    with pytest.raises(ValueError, match="V7"):
        _validate_training_data(frame)


def test_training_rejects_non_numeric_feature():
    from fraud_detection import _validate_training_data

    frame = _training_frame()
    frame.loc[0, "V2"] = "bad-value"
    with pytest.raises(ValueError, match="V2.*numeric"):
        _validate_training_data(frame)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_training_rejects_non_finite_features(bad_value):
    from fraud_detection import _validate_training_data

    frame = _training_frame()
    frame.loc[0, "Amount"] = bad_value
    with pytest.raises(ValueError, match="missing or infinite"):
        _validate_training_data(frame)


@pytest.mark.parametrize("labels", [[0, 2], [0, 0], ["fraud", "legit"]])
def test_training_rejects_invalid_target_labels(labels):
    from fraud_detection import _validate_training_data

    frame = _training_frame()
    frame["Class"] = [labels[i % len(labels)] for i in range(len(frame))]
    with pytest.raises(ValueError, match="Class"):
        _validate_training_data(frame)


def test_training_rejects_pre_scaled_amount_column():
    from fraud_detection import _validate_training_data

    frame = _training_frame().drop(columns=["Amount"])
    frame["scaled_amount"] = 0.0
    with pytest.raises(ValueError, match="raw 'Amount'"):
        _validate_training_data(frame)
