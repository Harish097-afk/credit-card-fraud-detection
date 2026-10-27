import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from fraud_detection import (
    AMOUNT_COLUMN,
    FEATURE_COLUMNS,
    _threshold_metrics,
    _validate_training_data,
    prepare_training_frame,
    save_artifacts,
    select_threshold,
)


def _frame(rows=12):
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(rng.normal(size=(rows, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame[AMOUNT_COLUMN] = np.arange(rows, dtype=float)
    frame["Class"] = [0, 1] * (rows // 2)
    return frame


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_prepare_training_frame_rejects_non_finite_values(bad_value):
    frame = _frame()
    frame.loc[0, "V3"] = bad_value
    with pytest.raises(ValueError, match="NaN or infinite"):
        prepare_training_frame(frame)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_validate_training_data_rejects_non_finite_target(bad_value):
    frame = _frame()
    frame["Class"] = frame["Class"].astype(float)
    frame.loc[0, "Class"] = bad_value
    with pytest.raises(ValueError, match="missing or infinite"):
        _validate_training_data(frame)


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.1, np.nan, np.inf, "bad"])
def test_save_artifacts_validates_threshold_before_writing(tmp_path, bad_threshold):
    model = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
    scaler = StandardScaler().fit([[0.0], [1.0]])
    model_path = tmp_path / "nested" / "model.joblib"
    scaler_path = tmp_path / "nested" / "scaler.joblib"

    with pytest.raises(ValueError, match="threshold"):
        save_artifacts(
            model,
            scaler,
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            threshold=bad_threshold,
            threshold_path=str(tmp_path / "nested" / "threshold.joblib"),
        )

    assert not model_path.exists()
    assert not scaler_path.exists()


def test_save_artifacts_creates_parent_directories(tmp_path):
    model = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
    scaler = StandardScaler().fit([[0.0], [1.0]])
    model_path = tmp_path / "artifacts" / "model.joblib"
    scaler_path = tmp_path / "artifacts" / "scaler.joblib"
    threshold_path = tmp_path / "config" / "threshold.joblib"

    save_artifacts(
        model,
        scaler,
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        threshold=0.4,
        threshold_path=str(threshold_path),
    )

    assert joblib.load(model_path) is not None
    assert joblib.load(scaler_path) is not None
    assert joblib.load(threshold_path) == pytest.approx(0.4)


@pytest.mark.parametrize("scores", [[-0.1, 0.5], [0.2, 1.1], [0.2, np.nan], [0.2, np.inf]])
def test_select_threshold_rejects_invalid_probabilities(scores):
    with pytest.raises(ValueError, match="Probabilities"):
        select_threshold(pd.Series([0, 1]), np.asarray(scores))


@pytest.mark.parametrize("scores", [[0.1], [0.1, 0.2, 0.3]])
def test_select_threshold_rejects_length_mismatch(scores):
    with pytest.raises(ValueError, match="equal lengths"):
        select_threshold(pd.Series([0, 1]), np.asarray(scores))


def test_threshold_metrics_rejects_invalid_threshold_and_single_class():
    scores = np.array([0.1, 0.9])
    with pytest.raises(ValueError, match="threshold"):
        _threshold_metrics(pd.Series([0, 1]), scores, 1.5)
    with pytest.raises(ValueError, match="both binary labels"):
        _threshold_metrics(pd.Series([0, 0]), scores, 0.5)
