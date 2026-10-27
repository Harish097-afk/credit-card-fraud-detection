import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from streamlit.testing.v1 import AppTest


def test_single_transaction_ui_renders_and_predicts(tmp_path, monkeypatch):
    rng = np.random.default_rng(12)
    features = rng.normal(size=(40, 29))
    labels = np.array([0, 1] * 20)
    model = LogisticRegression(max_iter=500).fit(features, labels)
    scaler = StandardScaler().fit(np.array([[0.0], [100.0], [500.0]]))

    model_path = tmp_path / "model.joblib"
    scaler_path = tmp_path / "scaler.joblib"
    threshold_path = tmp_path / "threshold.joblib"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(0.5, threshold_path)

    monkeypatch.setenv("FRAUD_MODEL_PATH", str(model_path))
    monkeypatch.setenv("FRAUD_SCALER_PATH", str(scaler_path))
    monkeypatch.setenv("FRAUD_THRESHOLD_PATH", str(threshold_path))

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not app.exception
    assert any("Single Transaction Predictor" in item.value for item in app.subheader)

    analyze_button = next(button for button in app.button if button.label == "Analyze Transaction")
    app = analyze_button.click().run()

    assert not app.exception
    prediction_messages = [
        item.value for item in list(app.success) + list(app.error)
        if "probability:" in item.value
    ]
    assert prediction_messages, "Clicking Analyze Transaction should display a prediction and probability."
