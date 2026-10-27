import numpy as np
import pytest

from review_policy import prioritize_reviews


def test_selects_highest_risk_rows_and_preserves_original_order():
    scores = np.array([0.1, 0.95, 0.2, 0.8, 0.4])
    selected = prioritize_reviews(scores, 40)
    assert selected.tolist() == [False, True, False, True, False]


def test_capacity_uses_ceiling_and_selects_at_least_one():
    assert prioritize_reviews([0.1, 0.2, 0.3], 34).sum() == 2
    assert prioritize_reviews([0.1, 0.2, 0.3], 0.01).sum() == 1


@pytest.mark.parametrize("capacity", [0, -1, 100.1, float("nan")])
def test_rejects_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        prioritize_reviews([0.1, 0.9], capacity)


def test_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        prioritize_reviews([0.2, 1.2], 50)
    with pytest.raises(ValueError):
        prioritize_reviews([0.2, float("nan")], 50)


def test_empty_input_returns_empty_mask():
    assert prioritize_reviews([], 0.5).size == 0

@pytest.mark.parametrize("capacity", [None, "bad", float("inf"), float("-inf")])
def test_rejects_non_numeric_or_non_finite_capacity(capacity):
    with pytest.raises(ValueError, match="capacity_percent"):
        prioritize_reviews([0.1, 0.9], capacity)


@pytest.mark.parametrize("scores", [[[0.1, 0.9]], [[0.1], [0.9]], ["bad", 0.9]])
def test_rejects_malformed_probability_inputs(scores):
    with pytest.raises(ValueError):
        prioritize_reviews(scores, 50)


def test_equal_scores_have_stable_input_order():
    selected = prioritize_reviews([0.5, 0.5, 0.5, 0.1], 50)
    assert selected.tolist() == [True, True, False, False]


def test_full_capacity_selects_every_row():
    assert prioritize_reviews([0.1, 0.9, 0.4], 100).tolist() == [True, True, True]

