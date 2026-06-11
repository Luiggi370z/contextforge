"""Retrieval protocols: factory selection, conformance, in-memory store."""

from __future__ import annotations

import uuid

import pytest

from app.retrieval.protocols import Embedder, Reranker, VectorStore


def test_factory_returns_hash_embedder_for_hash_backend(monkeypatch):
    from app.core.config import get_settings
    from app.retrieval.factory import get_embedder

    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    get_settings.cache_clear()
    embedder = get_embedder()
    get_settings.cache_clear()

    assert isinstance(embedder, Embedder)
    assert embedder.dim == 384
    assert embedder.name == "hash"


def test_factory_returns_lexical_reranker_for_lexical_backend(monkeypatch):
    from app.core.config import get_settings
    from app.retrieval.factory import get_reranker

    monkeypatch.setenv("RERANK_BACKEND", "lexical")
    get_settings.cache_clear()
    reranker = get_reranker()
    get_settings.cache_clear()

    assert isinstance(reranker, Reranker)
    assert reranker.name == "lexical"


@pytest.mark.asyncio
async def test_hash_embedder_embeds_deterministically(monkeypatch):
    from app.core.config import get_settings
    from app.retrieval.factory import get_embedder

    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    get_settings.cache_clear()
    embedder = get_embedder()
    get_settings.cache_clear()

    first = await embedder.embed(["hello world"])
    second = await embedder.embed(["hello world"])
    assert first == second
    assert len(first[0]) == 384


def test_qdrant_store_satisfies_vector_store_protocol():
    from app.retrieval.qdrant_store import QdrantStore

    store = QdrantStore()
    assert isinstance(store, VectorStore)
    assert store.name == "qdrant"


@pytest.mark.asyncio
async def test_in_memory_vector_store_roundtrip():
    from app.retrieval.vector_stores import InMemoryVectorStore

    store = InMemoryVectorStore(embedder_backend="hash")
    assert isinstance(store, VectorStore)

    document_id = uuid.uuid4()
    chunk_ids = [uuid.uuid4(), uuid.uuid4()]
    texts = [
        "Document: pto.md > Section: PTO\n\nFull-time staff accrue 20 PTO days.",
        "Document: pto.md > Section: Rollover\n\nUnused PTO rolls over up to 10 days.",
    ]
    bodies = [
        "Full-time staff accrue 20 PTO days.",
        "Unused PTO rolls over up to 10 days.",
    ]
    await store.upsert_chunks(document_id, chunk_ids, texts, bodies=bodies)

    results = await store.dense_search("how many PTO days", limit=2)
    assert results
    assert all(not record.content.startswith("Document: ") for record in results)
    assert results[0].chunk_id in chunk_ids
