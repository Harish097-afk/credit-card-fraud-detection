"""FastAPI bridge for the existing fraud detection model artifacts."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fraud_detection import (
    AMOUNT_COLUMN,
    FEATURE_COLUMNS,
    load_artifacts,
    load_threshold,
    prepare_inference_frame,
)
from review_policy import prioritize_reviews

MODEL_PATH = os.getenv("FRAUD_MODEL_PATH", "ensemble_model.joblib")
SCALER_PATH = os.getenv("FRAUD_SCALER_PATH", "scaler.joblib")
THRESHOLD_PATH = os.getenv("FRAUD_THRESHOLD_PATH", "threshold.joblib")

app = FastAPI(title="Fraudwatch API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "FRAUDWATCH_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)
try:
    default_threshold = load_threshold(THRESHOLD_PATH)
except (ValueError, OSError):
    default_threshold = 0.5


class TransactionInput(BaseModel):
    amount: float = Field(ge=0)
    features: dict[str, float]
    threshold: float = Field(default=default_threshold, ge=0, le=1)


class BatchInput(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=10000)
    policy: str = "capacity"
    capacity_percent: float = Field(default=0.5, gt=0, le=100)
    threshold: float = Field(default=default_threshold, ge=0, le=1)


def require_model() -> None:
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifacts are missing. Run "
                "'python train_model.py --data-path path/to/creditcard.csv' first."
            ),
        )


def score_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    require_model()
    try:
        prepared = prepare_inference_frame(frame, scaler)
        probabilities = np.asarray(model.predict_proba(prepared)[:, 1], dtype=float)
        if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
            raise ValueError("Model returned invalid probability scores.")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Model probabilities must be between 0 and 1.")
        return prepared, probabilities
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model is not None and scaler is not None,
        "default_threshold": default_threshold,
    }


@app.post("/predict")
def predict(payload: TransactionInput) -> dict[str, Any]:
    require_model()
    missing = [feature for feature in FEATURE_COLUMNS if feature not in payload.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")
    row = {**payload.features, AMOUNT_COLUMN: payload.amount}
    _, probabilities = score_frame(pd.DataFrame([row]))
    probability = float(probabilities[0])
    return {
        "probability": probability,
        "risk_score": round(probability * 100, 2),
        "prediction": int(probability >= payload.threshold),
        "threshold": payload.threshold,
    }


@app.post("/predict/batch")
def predict_batch(payload: BatchInput) -> dict[str, Any]:
    require_model()
    frame = pd.DataFrame(payload.rows)
    _, probabilities = score_frame(frame)
    if payload.policy == "capacity":
        priority = prioritize_reviews(probabilities, payload.capacity_percent)
    elif payload.policy == "threshold":
        priority = probabilities >= payload.threshold
    else:
        raise HTTPException(status_code=422, detail="policy must be 'capacity' or 'threshold'.")

    results = []
    for index, (probability, selected) in enumerate(zip(probabilities, priority), start=1):
        results.append({
            "row_number": index,
            "probability": float(probability),
            "risk_score": round(float(probability) * 100, 2),
            "priority": bool(selected),
        })
    results.sort(key=lambda row: row["probability"], reverse=True)
    return {
        "total": len(results),
        "review_count": int(np.asarray(priority, dtype=bool).sum()),
        "capacity_percent": payload.capacity_percent if payload.policy == "capacity" else None,
        "results": results,
    }
