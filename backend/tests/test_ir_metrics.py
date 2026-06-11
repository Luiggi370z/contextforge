"""Unit tests for pure-Python retrieval IR metrics (recall@k / MRR / nDCG@k)."""

from __future__ import annotations

import math

import pytest

from app.eval.ir_metrics import mrr, ndcg_at_k, recall_at_k, summarize_ir


def approx_exact(value: float):
    return pytest.approx(value, abs=1e-9)


def test_recall_hit_at_rank_one():
    assert recall_at_k(["a", "b", "c"], {"a"}, k=5) == 1.0


def test_recall_miss():
    assert recall_at_k(["b", "c"], {"a"}, k=5) == 0.0


def test_recall_respects_k():
    assert recall_at_k(["b", "c", "a"], {"a"}, k=2) == 0.0
    assert recall_at_k(["b", "c", "a"], {"a"}, k=3) == 1.0


def test_mrr_rank_one_is_one():
    assert mrr(["a", "b"], {"a"}) == 1.0


def test_mrr_rank_three_is_one_third():
    assert mrr(["x", "y", "a"], {"a"}) == approx_exact(1 / 3)


def test_mrr_no_hit_is_zero():
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_ndcg_perfect_when_relevant_first():
    assert ndcg_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0


def test_ndcg_discounts_lower_ranks():
    expected = (1 / math.log2(3)) / 1.0
    assert ndcg_at_k(["x", "a", "y"], {"a"}, k=3) == approx_exact(expected)


def test_summarize_ir_averages_rows():
    rows = [
        {"retrieved": ["a", "b"], "relevant": {"a"}},
        {"retrieved": ["x", "y"], "relevant": {"a"}},
    ]
    summary = summarize_ir(rows, ks=(5,))
    assert summary["recall@5"] == approx_exact(0.5)
    assert summary["mrr"] == approx_exact(0.5)
