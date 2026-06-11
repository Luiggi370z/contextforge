"""Pure-Python retrieval IR metrics: recall@k, MRR, nDCG@k.

Hand-rolled (no ``pytrec_eval``/``ir-measures`` C-extension) so the eval gate
stays deterministic and dependency-light. Relevance is binary and keyed on
document id (the golden set's stable ``reference_doc`` filename), so these are
document-level retrieval metrics.

Each input is a ranked list of retrieved doc ids (best first) plus the set of
ids considered relevant for that query.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant ids that appear in the top-``k`` retrieved ids."""
    if not relevant:
        return 0.0
    topk = set(retrieved[:k])
    hits = len(topk & relevant)
    return hits / len(relevant)


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant id (0.0 if none retrieved)."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG@k. IDCG is computed for the ideal binary ranking."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for index, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(index + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def summarize_ir(
    rows: Iterable[dict],
    *,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    """Average recall@k (for each k), MRR, and nDCG@max(k) across rows.

    Each row is ``{"retrieved": [doc_id, ...], "relevant": {doc_id, ...}}``.
    Rows whose ``relevant`` set is empty (e.g. abstain rows) are EXCLUDED from
    the denominator entirely — the returned means are over scored rows only, not
    over all input rows. Callers that need the row count should track it themselves.
    """
    scored = [row for row in rows if row.get("relevant")]
    if not scored:
        return {}
    summary: dict[str, float] = {}
    for k in ks:
        recalls = [recall_at_k(row["retrieved"], set(row["relevant"]), k) for row in scored]
        summary[f"recall@{k}"] = round(sum(recalls) / len(recalls), 4)
    mrrs = [mrr(row["retrieved"], set(row["relevant"])) for row in scored]
    summary["mrr"] = round(sum(mrrs) / len(mrrs), 4)
    max_k = max(ks)
    ndcgs = [ndcg_at_k(row["retrieved"], set(row["relevant"]), max_k) for row in scored]
    summary[f"ndcg@{max_k}"] = round(sum(ndcgs) / len(ndcgs), 4)
    return summary
