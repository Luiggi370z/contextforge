import os

import pytest

os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("LLM_PROVIDER", "heuristic")
os.environ.setdefault("APP_ENV", "test")


@pytest.fixture(autouse=True)
def _clear_embedding_cache():
    from app.retrieval import embeddings

    embeddings._sentence_model.cache_clear()
    yield
    embeddings._sentence_model.cache_clear()


@pytest.fixture(autouse=True)
def _clear_llm_provider_cache():
    from app.llm.providers import reset_llm_provider_cache

    reset_llm_provider_cache()
    yield
    reset_llm_provider_cache()
