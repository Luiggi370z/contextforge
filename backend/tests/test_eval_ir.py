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

# (use_sparse, use_rerank) per retrieval config, mirroring EvalService.CONFIG_LEGS.
_CONFIGS: dict[str, tuple[bool, bool]] = {
    "dense_only": (False, False),
    "hybrid": (True, False),
    "hybrid_rerank": (True, True),
}


async def _ir_for_config(
    corpus: InMemoryCorpus, *, use_sparse: bool, use_rerank: bool
) -> dict[str, float]:
    rows = [row for row in load_golden() if not row.expect_abstain]
    ir_rows = []
    for row in rows:
        chunks = await hybrid_retrieve_in_memory(
            corpus, row.question, use_sparse=use_sparse, use_rerank=use_rerank
        )
        # Resolve filename via the corpus, not chunk.metadata: dense-only hits carry
        # no metadata (VectorRecord has none), so reading metadata would score every
        # dense-only row as a miss. filename_for() is leg-independent.
        retrieved_docs = [corpus.filename_for(chunk.chunk_id) for chunk in chunks]
        ir_rows.append({"retrieved": retrieved_docs, "relevant": row.relevant_set})
    return summarize_ir(ir_rows, ks=(1, 3, 5))


@pytest.mark.asyncio
async def test_ir_metrics_on_golden_set(corpus: InMemoryCorpus) -> None:
    """recall@5/MRR/nDCG@5 over the golden set meet a quality floor."""
    summary = await _ir_for_config(corpus, use_sparse=True, use_rerank=True)
    print("\nIR metrics (golden, hybrid+rerank):", summary)

    assert summary["recall@5"] >= 0.9, summary
    assert summary["mrr"] >= 0.7, summary
    assert summary["ndcg@5"] >= 0.7, summary


@pytest.mark.asyncio
async def test_retrieval_quality_improves_across_configs(corpus: InMemoryCorpus) -> None:
    """The full hybrid+rerank pipeline beats dense-only on the hard corpus.

    The overlapping PDF distractors make this lift real and measurable. The
    honest finding on this corpus (MiniLM-384 + BM25) is that *naive* RRF hybrid
    does NOT by itself beat dense retrieval — the cross-encoder reranker is the
    lever that moves the numbers. So the gate asserts the robust, true relations:
    the end-to-end pipeline beats dense-only, and rerank improves on plain
    hybrid. It deliberately does NOT assert ``hybrid > dense_only``, because that
    is not what the data shows.
    """
    summaries = {
        name: await _ir_for_config(corpus, use_sparse=us, use_rerank=ur)
        for name, (us, ur) in _CONFIGS.items()
    }
    for name, summary in summaries.items():
        print(f"\nIR metrics ({name}):", summary)

    dense = summaries["dense_only"]
    hybrid = summaries["hybrid"]
    rerank = summaries["hybrid_rerank"]

    # End-to-end pipeline must beat dense-only, and rerank must lift plain hybrid.
    assert rerank["mrr"] > dense["mrr"], summaries
    assert rerank["recall@1"] >= dense["recall@1"], summaries
    assert rerank["mrr"] > hybrid["mrr"], summaries
    assert rerank["recall@1"] >= hybrid["recall@1"], summaries
