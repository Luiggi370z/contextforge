"""Embedding providers — hash backend for fast tests, sentence-transformers for real use.

TODO(retrieval-backend): Same vectors feed Qdrant upsert today; in postgres mode, write to
``chunks.embedding`` (pgvector) at ingest time via app/ingestion/service.py.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np
import structlog

from app.core.config import get_settings

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # optional ``ml`` extra — hash backend does not need it
    SentenceTransformer = None  # type: ignore[misc, assignment]

log = structlog.get_logger(__name__)
_DIM = 384


def _hash_embed(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    vec = rng.standard_normal(_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec.tolist()


@lru_cache(maxsize=1)
def _sentence_model():
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is required for embedding_backend != 'hash'. "
            "Install with: uv sync --extra ml"
        )
    settings = get_settings()
    log.info("loading_embedding_model", model=settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    settings = get_settings()
    if settings.embedding_backend == "hash":
        return [_hash_embed(t) for t in texts]
    model = _sentence_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]
