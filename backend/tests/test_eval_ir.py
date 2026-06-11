"""Deterministic document-level IR metrics over the golden set (offline).

Runs under ``pytest -m eval``. Reuses the in-memory corpus + real embed/rerank
stack from the existing golden gate, but instead of asserting pass/fail per row
it computes recall@k / MRR / nDCG@k keyed on the stable ``reference_doc``
filename and asserts a monotonic-quality floor. Prints the numbers so the run
doubles as the "how good is retrieval" report.
"""

from __future__ import annotations

import pytest

from app.eval.golden_loader import load_golden
from app.eval.harness import InMemoryCorpus, hybrid_retrieve_in_memory
from app.eval.ir_metrics import summarize_ir

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
async def test_ir_metrics_on_golden_set(corpus: InMemoryCorpus) -> None:
    """recall@5/MRR/nDCG@5 over the golden set meet a quality floor."""
    rows = load_golden()
    ir_rows = []
    for row in rows:
        if row.expect_abstain:
            continue
        chunks = await hybrid_retrieve_in_memory(corpus, row.question)
        retrieved_docs = [
            str((chunk.metadata or {}).get("filename") or "") for chunk in chunks
        ]
        ir_rows.append({"retrieved": retrieved_docs, "relevant": {row.reference_doc}})

    summary = summarize_ir(ir_rows, ks=(1, 3, 5))
    print("\nIR metrics (golden, hybrid+rerank):", summary)

    assert summary["recall@5"] >= 0.9, summary
    assert summary["mrr"] >= 0.7, summary
    assert summary["ndcg@5"] >= 0.7, summary
