"""Deterministic, dependency-light evaluation toolkit (shippable).

Bundles the in-memory retrieval harness, the labelled golden set, the reference
corpus, and the pure-Python IR metrics so the ``/v1/eval`` endpoint and the
``pytest -m eval`` gate run with no Qdrant, no live API, and no repo-root files
on the path. Everything here lives under ``app`` so it ships in the wheel /
container image rather than depending on ``tests/`` or repo-root ``eval/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.eval.harness import InMemoryCorpus

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"

# Loaders the eval corpus exercises. Markdown holds the canonical golden answers;
# the verbose PDFs are overlapping distractors that the retriever must rank below
# the precise policy for a factual question.
CORPUS_GLOBS = ("*.md", "*.pdf", "*.txt")


def build_reference_corpus() -> InMemoryCorpus:
    """Ingest every bundled corpus file (md + pdf + txt) into a fresh corpus.

    Single source of truth for both the ``pytest -m eval`` fixture and the
    ``/v1/eval`` service so they always evaluate against the same documents.
    """
    from app.eval.harness import InMemoryCorpus

    corpus = InMemoryCorpus()
    paths = sorted(path for glob in CORPUS_GLOBS for path in CORPUS_DIR.glob(glob))
    for path in paths:
        corpus.ingest_file(path)
    return corpus


__all__ = ["CORPUS_DIR", "CORPUS_GLOBS", "GOLDEN_PATH", "build_reference_corpus"]
