"""Text embedding providers.

- ``hash`` backend: deterministic, dependency-free. Used by tests so we do not
  load a SentenceTransformer model into memory.
- ``sentence-transformers`` backend (the default for ingest + retrieval): loads
  the configured model (e.g. ``all-MiniLM-L6-v2``) once, caches it, and runs
  ``encode`` off the event loop so async callers (FastAPI handlers, graph
  nodes) do not block on CPU work.
"""

from __future__ import annotations

import asyncio
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
    """Synchronous embedding (sync ingest scripts, tests, eval harness).

    Async callers should use :func:`embed_texts_async` so the model.encode call
    does not block the event loop.
    """
    if not texts:
        return []
    settings = get_settings()
    if settings.embedding_backend == "hash":
        return [_hash_embed(text) for text in texts]
    model = _sentence_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


async def embed_texts_async(texts: list[str]) -> list[list[float]]:
    """Async-safe wrapper: hashing is cheap inline, encode runs in a worker thread."""
    if not texts:
        return []
    settings = get_settings()
    if settings.embedding_backend == "hash":
        return [_hash_embed(text) for text in texts]
    model = _sentence_model()
    vectors = await asyncio.to_thread(model.encode, texts, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]
