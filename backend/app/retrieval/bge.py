"""Local BGE-M3: one model emits dense (1024) + sparse from a single load.

FastEmbed does not ship BGE-M3, so this uses FlagEmbedding's BGEM3FlagModel —
the one model that feeds both Qdrant named vectors in the fully-offline path.
``FlagEmbedding`` is a heavy optional dependency (the ``ml`` extra); import is
guarded so the default install and the eval gate never need it.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

import structlog

try:
    from FlagEmbedding import BGEM3FlagModel  # pyright: ignore[reportMissingImports]
except ImportError:  # optional ``ml`` extra
    BGEM3FlagModel = None  # type: ignore[misc, assignment]

log = structlog.get_logger(__name__)

_BGE_M3_MODEL = "BAAI/bge-m3"
SparseVectorTuple = tuple[list[int], list[float]]


@lru_cache(maxsize=1)
def _bge_model():
    if BGEM3FlagModel is None:
        raise ImportError(
            "FlagEmbedding is required for BGE-M3. Install with: uv sync --extra ml"
        )
    log.info("loading_bge_m3", model=_BGE_M3_MODEL)
    return BGEM3FlagModel(_BGE_M3_MODEL, use_fp16=True)


def _encode(texts: list[str], *, dense: bool, sparse: bool) -> dict:
    return _bge_model().encode(
        texts, return_dense=dense, return_sparse=sparse, return_colbert_vecs=False
    )


def bge_dense_sync(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    out = _encode(texts, dense=True, sparse=False)
    return [vec.tolist() for vec in out["dense_vecs"]]


async def bge_dense_async(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return await asyncio.to_thread(bge_dense_sync, texts)


def _lexical_to_tuple(weights: dict) -> SparseVectorTuple:
    """Convert BGE-M3 lexical_weights {token_id: weight} to Qdrant (indices, values)."""
    indices = [int(token_id) for token_id in weights]
    values = [float(weight) for weight in weights.values()]
    return indices, values


def bge_sparse_sync(texts: list[str]) -> list[SparseVectorTuple]:
    if not texts:
        return []
    out = _encode(texts, dense=False, sparse=True)
    return [_lexical_to_tuple(weights) for weights in out["lexical_weights"]]


async def bge_sparse_async(texts: list[str]) -> list[SparseVectorTuple]:
    if not texts:
        return []
    return await asyncio.to_thread(bge_sparse_sync, texts)
