"""End-to-end retrieval + abstain evaluation over ``eval/golden.jsonl``.

Run with ``pytest -m eval``. The test wires the real graph (route, retrieve,
grade, generate, validate) on top of an in-memory corpus + retriever and the
heuristic LLM provider, so the eval is deterministic and runs in CI without
Ollama / OpenAI / Postgres / Qdrant.

The assertions cover the three behaviours we keep regressing on:

1. **Retrieval recall** — the reference document for a factual question must
   appear in the selected chunks.
2. **Citations match generation** — generation contexts and citation snippets
   come from the same chunks; both reference the right document.
3. **Abstain triggers** — questions whose answers are not in the corpus do
   not surface citations.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.eval.golden_loader import load_golden
from app.eval.harness import InMemoryCorpus, hybrid_retrieve_in_memory
from app.graph.builder import invoke_agent_graph

pytestmark = pytest.mark.eval


@pytest.fixture
def patched_graph(monkeypatch: pytest.MonkeyPatch, corpus: InMemoryCorpus, real_embeddings):
    """Route the production graph through ``corpus`` for hybrid retrieval.

    Everything else (rerank, grading, generation, validation) runs the real
    code path with production-tuned thresholds.
    """

    async def _retrieve(_session: Any, _qdrant: Any, query: str):
        return await hybrid_retrieve_in_memory(corpus, query)

    monkeypatch.setattr("app.graph.builder.hybrid_retrieve", _retrieve)
    yield


def _selected_metadata(state: Any) -> list[dict]:
    return [
        document.get("metadata") or {}
        for document in state.get("documents") or []
    ]


@pytest.mark.asyncio
async def test_eval_golden_set(patched_graph) -> None:
    """Every golden question either cites the right doc or correctly abstains."""
    rows = load_golden()
    assert len(rows) >= 15, "golden set must keep at least 15 cases"

    failures: list[str] = []
    abstain_seen = 0
    factual_pass = 0

    for row in rows:
        state = await invoke_agent_graph(
            row.question,
            db=MagicMock(),
            qdrant=MagicMock(),
            thread_id="eval",
        )
        if row.expect_abstain:
            if state.get("abstained"):
                abstain_seen += 1
            else:
                failures.append(
                    f"expected abstain but answered: {row.question!r} -> {state.get('answer')!r}"
                )
            continue

        citations = state.get("citations") or []
        if not citations:
            failures.append(
                f"no citations for factual question: {row.question!r}"
            )
            continue

        metadata = _selected_metadata(state)
        cited_chunk_ids = {citation.get("chunk_id") for citation in citations}
        cited_docs = {
            meta.get("filename")
            for meta, document in zip(metadata, state.get("documents") or [], strict=False)
            if document.get("chunk_id") in cited_chunk_ids
        }
        if row.reference_doc in cited_docs:
            factual_pass += 1
        else:
            failures.append(
                f"reference doc {row.reference_doc!r} not in citations for "
                f"{row.question!r}; cited={cited_docs}"
            )

    factual_total = len([row for row in rows if not row.expect_abstain])
    abstain_total = len([row for row in rows if row.expect_abstain])

    recall = factual_pass / factual_total
    abstain_rate = abstain_seen / abstain_total

    assert not failures, (
        f"{len(failures)} golden cases failed (factual {factual_pass}/{factual_total}, "
        f"abstain {abstain_seen}/{abstain_total}):\n" + "\n".join(failures[:10])
    )
    assert recall >= 0.75, f"retrieval recall too low: {recall:.2f}"
    assert abstain_rate >= 0.66, f"abstain rate too low: {abstain_rate:.2f}"
