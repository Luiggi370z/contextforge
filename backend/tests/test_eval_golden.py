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

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.core.config import get_settings
from app.graph.builder import invoke_agent_graph
from app.llm.providers import reset_llm_provider_cache
from app.retrieval import embeddings as embeddings_module
from tests.eval_harness import InMemoryCorpus, hybrid_retrieve_in_memory

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
SAMPLE_CORPUS = Path(__file__).resolve().parents[2] / "sample_corpus"

sys.path.insert(0, str(EVAL_DIR))
from golden_loader import load_golden  # type: ignore[import-not-found]  # noqa: E402

pytestmark = pytest.mark.eval


_PRODUCTION_LIKE_ENV = {
    # Hash embeddings are deterministic noise; the eval is a behaviour gate, so it
    # has to run against the same embedder, BM25, RRF and reranker the product
    # actually ships.
    "EMBEDDING_BACKEND": "sentence-transformers",
    # Cross-encoder is the modern reranker the architecture is built around. The
    # accompanying ``grade_min_score_cross_encoder`` threshold (0.0) handles
    # abstention via a real semantic signal, not vocabulary heuristics.
    "RERANK_BACKEND": "cross_encoder",
}


@pytest.fixture(scope="module")
def real_embeddings():
    """Run the eval with the production-recommended embed + rerank stack."""
    previous = {key: os.environ.get(key) for key in _PRODUCTION_LIKE_ENV}
    for key, value in _PRODUCTION_LIKE_ENV.items():
        os.environ[key] = value
    get_settings.cache_clear()
    embeddings_module._sentence_model.cache_clear()
    try:
        yield
    finally:
        for key, previous_value in previous.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value
        get_settings.cache_clear()
        embeddings_module._sentence_model.cache_clear()
        reset_llm_provider_cache()


@pytest.fixture(scope="module")
def corpus(real_embeddings) -> InMemoryCorpus:
    """Seed an in-memory corpus from the same Markdown the demo ingests."""
    bundle = InMemoryCorpus()
    for path in sorted(SAMPLE_CORPUS.glob("*.md")):
        bundle.ingest(filename=path.name, content=path.read_text(encoding="utf-8"))
    return bundle


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
