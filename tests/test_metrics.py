"""Tests for Recall@K and NDCG@K computation."""

import math

import numpy as np
import pytest


def compute_recall_at_k(predicted: list[int], actual: set[int], k: int) -> float:
    """Compute STANDARD Recall@K for a single user: hits / |relevant|.

    Denominator is the user's full relevant count (NOT min(|relevant|, K)),
    so Recall@K is monotone non-decreasing in K. Mirrors the evaluator fix.
    """
    pred_k = predicted[:k]
    hits = sum(1.0 for item in pred_k if item in actual)
    return hits / len(actual) if actual else 0.0


def compute_ndcg_at_k(predicted: list[int], actual: set[int], k: int) -> float:
    """Compute NDCG@K for a single user."""
    pred_k = predicted[:k]
    dcg = sum(
        (1.0 if item in actual else 0.0) / math.log2(i + 2)
        for i, item in enumerate(pred_k)
    )
    ideal_len = min(len(actual), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
    return dcg / idcg if idcg > 0 else 0.0


class TestRecall:
    def test_perfect_recall(self):
        predicted = [0, 1, 2, 3, 4]
        actual = {0, 1, 2}
        assert compute_recall_at_k(predicted, actual, 5) == pytest.approx(1.0)

    def test_zero_recall(self):
        predicted = [5, 6, 7, 8, 9]
        actual = {0, 1, 2}
        assert compute_recall_at_k(predicted, actual, 5) == pytest.approx(0.0)

    def test_partial_recall(self):
        predicted = [0, 5, 1, 6, 7]
        actual = {0, 1, 2}
        # At K=5: 2 hits out of 3 relevant
        assert compute_recall_at_k(predicted, actual, 5) == pytest.approx(2.0 / 3.0)

    def test_recall_at_smaller_k(self):
        predicted = [0, 5, 1, 6, 7]
        actual = {0, 1, 2}
        # At K=2: 1 hit, standard recall = 1 / |relevant| = 1/3
        assert compute_recall_at_k(predicted, actual, 2) == pytest.approx(1.0 / 3.0)

    def test_recall_monotone_in_k(self):
        # Standard recall must never decrease as K grows (the property the
        # capped denominator violated on high-n_rel datasets).
        predicted = [0, 9, 1, 8, 2, 7, 6, 5, 4, 3]
        actual = {0, 1, 2, 10, 11}  # 5 relevant, 3 reachable in this list
        vals = [compute_recall_at_k(predicted, actual, k) for k in (1, 2, 5, 10)]
        assert vals == sorted(vals), f"recall not monotone in K: {vals}"


class TestNDCG:
    def test_perfect_ndcg(self):
        predicted = [0, 1, 2]
        actual = {0, 1, 2}
        assert compute_ndcg_at_k(predicted, actual, 3) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        predicted = [5, 6, 7]
        actual = {0, 1, 2}
        assert compute_ndcg_at_k(predicted, actual, 3) == pytest.approx(0.0)

    def test_ndcg_ordering_matters(self):
        actual = {0}
        # Hit at position 1 (best for single relevant item)
        ndcg_first = compute_ndcg_at_k([0, 1, 2], actual, 3)
        # Hit at position 3 (worst)
        ndcg_last = compute_ndcg_at_k([1, 2, 0], actual, 3)
        assert ndcg_first > ndcg_last

    def test_ndcg_known_value(self):
        predicted = [0, 5, 1]
        actual = {0, 1}
        # DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.0 + 0 + 0.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309...
        dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected = dcg / idcg
        assert compute_ndcg_at_k(predicted, actual, 3) == pytest.approx(expected, rel=1e-4)
