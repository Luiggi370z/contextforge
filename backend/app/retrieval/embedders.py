"""Concrete Embedder implementations (delegate to app.retrieval.embeddings)."""

from __future__ import annotations


class HashEmbedder:
    """Deterministic, dependency-free embedder for tests/offline runs."""

    name = "hash"

    def __init__(self) -> None:
        from app.retrieval.embeddings import embedding_dim

        self.dim = embedding_dim()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from app.retrieval.embeddings import _hash_embed

        return [_hash_embed(text) for text in texts]


class SentenceTransformerEmbedder:
    """Sentence-Transformers embedder; encode runs off the event loop."""

    name = "sentence-transformers"

    def __init__(self) -> None:
        from app.retrieval.embeddings import embedding_dim

        self.dim = embedding_dim()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        from app.retrieval.embeddings import _sentence_model

        if not texts:
            return []
        model = _sentence_model()
        vectors = await asyncio.to_thread(model.encode, texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


class BgeM3Embedder:
    """Local BGE-M3 dense embedder (1024-dim). Sparse comes from sparse.py / bge.py."""

    name = "bge-m3"

    def __init__(self) -> None:
        from app.retrieval.embeddings import embedding_dim

        self.dim = embedding_dim()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from app.retrieval.bge import bge_dense_async

        return await bge_dense_async(texts)
