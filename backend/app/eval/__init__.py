"""Deterministic, dependency-light evaluation toolkit (shippable).

Bundles the in-memory retrieval harness, the labelled golden set, the reference
corpus, and the pure-Python IR metrics so the ``/v1/eval`` endpoint and the
``pytest -m eval`` gate run with no Qdrant, no live API, and no repo-root files
on the path. Everything here lives under ``app`` so it ships in the wheel /
container image rather than depending on ``tests/`` or repo-root ``eval/``.
"""

from __future__ import annotations

from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
GOLDEN_PATH = Path(__file__).resolve().parent / "golden.jsonl"

__all__ = ["CORPUS_DIR", "GOLDEN_PATH"]
