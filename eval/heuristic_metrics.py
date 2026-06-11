"""Lexical eval metrics when RAGAS / LLM judges are unavailable."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.eval.golden_loader import GoldenRow


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


@dataclass(frozen=True)
class HeuristicScores:
    answer_overlap: float
    context_overlap: float
    abstain_correct: bool | None


def score_row(
    row: GoldenRow,
    *,
    answer: str,
    contexts: list[str],
    abstained: bool,
) -> HeuristicScores:
    """Score one row with simple token overlap heuristics."""
    truth_tokens = _tokenize(row.ground_truth)
    answer_tokens = _tokenize(answer)
    answer_overlap = (
        len(truth_tokens & answer_tokens) / len(truth_tokens) if truth_tokens else 0.0
    )

    context_blob = " ".join(contexts).lower()
    context_hits = sum(1 for token in truth_tokens if token in context_blob)
    context_overlap = context_hits / len(truth_tokens) if truth_tokens else 0.0

    abstain_correct: bool | None = None
    if row.expect_abstain:
        abstain_correct = abstained or "don't have enough information" in answer.lower()

    return HeuristicScores(
        answer_overlap=round(answer_overlap, 4),
        context_overlap=round(context_overlap, 4),
        abstain_correct=abstain_correct,
    )


def summarize_heuristic(scores: list[HeuristicScores]) -> dict[str, float]:
    """Aggregate heuristic scores across rows."""
    if not scores:
        return {}
    return {
        "answer_overlap_mean": round(
            sum(score.answer_overlap for score in scores) / len(scores), 4
        ),
        "context_overlap_mean": round(
            sum(score.context_overlap for score in scores) / len(scores), 4
        ),
        "abstain_accuracy": round(
            sum(1 for score in scores if score.abstain_correct is True)
            / max(1, sum(1 for score in scores if score.abstain_correct is not None)),
            4,
        ),
    }
