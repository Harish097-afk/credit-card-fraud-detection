"""Capacity-based transaction review helpers."""
from __future__ import annotations

import numpy as np


def prioritize_reviews(probabilities, capacity_percent: float = 0.5) -> np.ndarray:
    """Return a boolean mask selecting the highest-risk rows within capacity.

    The returned mask is aligned to the original input order. Capacity is a
    percentage of rows, not a fraud-probability threshold.
    """
    try:
        capacity = float(capacity_percent)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "capacity_percent must be a finite number greater than 0 and at most 100."
        ) from exc
    if not np.isfinite(capacity) or not 0 < capacity <= 100:
        raise ValueError(
            "capacity_percent must be a finite number greater than 0 and at most 100."
        )

    try:
        scores = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Probabilities must be numeric values.") from exc
    if scores.ndim != 1:
        raise ValueError("Probabilities must be a one-dimensional array.")
    if scores.size == 0:
        return np.zeros(0, dtype=bool)
    if not np.isfinite(scores).all():
        raise ValueError("Probabilities must be finite numbers.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    review_count = max(1, int(np.ceil(len(scores) * capacity / 100)))
    ranked_indices = np.argsort(-scores, kind="stable")
    selected = np.zeros(len(scores), dtype=bool)
    selected[ranked_indices[:review_count]] = True
    return selected
