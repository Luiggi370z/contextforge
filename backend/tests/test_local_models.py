"""BGE-M3 local embedder: correct dim + factory selection."""

from __future__ import annotations

import pytest

from app.retrieval.protocols import Embedder


def test_factory_selects_bge_m3(monkeypatch):
    from app.core.config import get_settings
    from app.retrieval.factory import get_embedder

    monkeypatch.setenv("EMBEDDING_BACKEND", "bge-m3")
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    get_settings.cache_clear()
    embedder = get_embedder()
    get_settings.cache_clear()

    assert isinstance(embedder, Embedder)
    assert embedder.name == "bge-m3"
    assert embedder.dim == 1024


@pytest.mark.asyncio
async def test_bge_m3_embeds_to_configured_dim(monkeypatch):
    pytest.importorskip("FlagEmbedding")
    from app.core.config import get_settings
    from app.retrieval.factory import get_embedder

    monkeypatch.setenv("EMBEDDING_BACKEND", "bge-m3")
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    get_settings.cache_clear()
    embedder = get_embedder()
    try:
        vectors = await embedder.embed(["20 PTO days per year"])
        assert len(vectors) == 1
        assert len(vectors[0]) == 1024
    finally:
        get_settings.cache_clear()
