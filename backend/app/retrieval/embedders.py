"""Concrete Embedder implementations (delegate to app.retrieval.embeddings)."""

from __future__ import annotations

from app.retrieval.embeddings import _DIM, _hash_embed


class HashEmbedder:
    """Deterministic, dependency-free embedder for tests/offline runs."""

    name = "hash"
    dim = _DIM

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embed(text) for text in texts]


class SentenceTransformerEmbedder:
    """Sentence-Transformers embedder; encode runs off the event loop."""

    name = "sentence-transformers"
    dim = _DIM

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        from app.retrieval.embeddings import _sentence_model

        if not texts:
            return []
        model = _sentence_model()
        vectors = await asyncio.to_thread(model.encode, texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]
