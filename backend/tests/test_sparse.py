"""FastEmbed BM25 sparse encoder produces aligned (indices, values)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastembed")

from app.retrieval.sparse import embed_sparse_async


@pytest.mark.asyncio
async def test_sparse_embed_returns_aligned_indices_values():
    vectors = await embed_sparse_async(["20 PTO days per calendar year"])
    assert len(vectors) == 1
    indices, values = vectors[0]
    assert len(indices) == len(values)
    assert len(indices) > 0
    assert all(isinstance(i, int) for i in indices)
    assert all(isinstance(v, float) for v in values)
