"""Tests for eval/golden.jsonl helpers (no live API or RAGAS required)."""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from golden_loader import GoldenRow, load_golden  # type: ignore[import-not-found]  # noqa: E402
from heuristic_metrics import (  # type: ignore[import-not-found]  # noqa: E402
    score_row,
    summarize_heuristic,
)


def test_load_golden_has_fifteen_rows():
    rows = load_golden()
    assert len(rows) >= 15
    assert all(isinstance(row, GoldenRow) for row in rows)


def test_golden_abstain_row_flagged():
    rows = load_golden()
    abstain_rows = [row for row in rows if row.expect_abstain]
    assert len(abstain_rows) >= 1


def test_heuristic_scores_overlap():
    row = GoldenRow(
        question="PTO?",
        ground_truth="20 PTO days per calendar year",
        reference_doc="policy_pto.md",
    )
    scores = score_row(
        row,
        answer="Full-time staff receive 20 PTO days per calendar year.",
        contexts=["Full-time employees accrue 20 PTO days per calendar year."],
        abstained=False,
    )
    assert scores.answer_overlap > 0.3
    assert scores.context_overlap > 0.3


def test_heuristic_abstain_detection():
    row = GoldenRow(
        question="Unknown?",
        ground_truth="Not documented",
        reference_doc="none",
        expect_abstain=True,
    )
    scores = score_row(
        row,
        answer="I don't have enough information in the indexed documents to answer that.",
        contexts=[],
        abstained=True,
    )
    assert scores.abstain_correct is True


def test_summarize_heuristic_empty():
    assert summarize_heuristic([]) == {}
