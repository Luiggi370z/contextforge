"""Citations must never leak the structural embedding prefix.

The dense leg embeds prefixed text (``Document: … > Section: …\n\n{body}``)
but the Qdrant payload — and therefore every citation snippet — must contain
only the clean body.
"""

from __future__ import annotations

import uuid

import pytest

from app.retrieval.qdrant_store import QdrantStore


class _FakeAsyncClient:
    """Captures the points handed to upsert so we can inspect the payload."""

    def __init__(self) -> None:
        self.upserted_points: list = []

    async def collection_exists(self, collection_name):  # noqa: ANN001
        return True

    async def create_collection(self, **kwargs):  # noqa: ANN003
        return True

    async def upsert(self, *, collection_name, points):  # noqa: ANN001
        self.upserted_points = points


@pytest.mark.asyncio
async def test_qdrant_payload_stores_clean_body_not_prefixed_text(monkeypatch):
    """upsert_chunks embeds prefixed text but stores the clean body in payload."""

    captured_texts: list[str] = []

    async def _fake_embed(texts):
        captured_texts.extend(texts)
        return [[0.0] * 384 for _ in texts]

    async def _fake_sparse(texts):
        # Sparse vectors are irrelevant to this payload assertion; stub them so the
        # test needs no fastembed model (keeps the CI `dev` extra free of `ml`).
        return [([0], [0.0]) for _ in texts]

    monkeypatch.setattr("app.retrieval.qdrant_store.embed_texts_async", _fake_embed)
    monkeypatch.setattr("app.retrieval.qdrant_store.embed_sparse_async", _fake_sparse)

    store = QdrantStore()
    fake_client = _FakeAsyncClient()
    store._client = fake_client  # type: ignore[attr-defined]

    document_id = uuid.uuid4()
    chunk_ids = [uuid.uuid4()]
    prefixed_texts = ["Document: pto.md > Section: PTO\n\n20 PTO days per year."]
    clean_bodies = ["20 PTO days per year."]

    await store.upsert_chunks(
        document_id,
        chunk_ids,
        prefixed_texts,
        bodies=clean_bodies,
    )

    payload = fake_client.upserted_points[0].payload
    assert payload["content"] == "20 PTO days per year."
    assert not payload["content"].startswith("Document: ")

    # Complementary invariant: the dense vector must be computed from the
    # PREFIXED text (retrieval geometry), even though the stored payload is clean.
    assert captured_texts[0].startswith("Document: ")
    assert "20 PTO days per year." in captured_texts[0]
