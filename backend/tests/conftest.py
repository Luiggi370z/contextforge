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


_PRODUCTION_LIKE_ENV = {
    "EMBEDDING_BACKEND": "sentence-transformers",
    "RERANK_BACKEND": "cross_encoder",
}


@pytest.fixture(scope="module")
def real_embeddings():
    """Production-recommended embed + rerank stack for eval-marked tests."""
    from app.core.config import get_settings
    from app.llm.providers import reset_llm_provider_cache
    from app.retrieval import embeddings as embeddings_module

    previous = {key: os.environ.get(key) for key in _PRODUCTION_LIKE_ENV}
    for key, value in _PRODUCTION_LIKE_ENV.items():
        os.environ[key] = value
    get_settings.cache_clear()
    embeddings_module._sentence_model.cache_clear()
    try:
        yield
    finally:
        for key, previous_value in previous.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value
        get_settings.cache_clear()
        embeddings_module._sentence_model.cache_clear()
        reset_llm_provider_cache()


@pytest.fixture(scope="module")
def corpus(real_embeddings):
    """Seed an in-memory corpus from the bundled reference docs (md + pdf + txt)."""
    from app.eval import build_reference_corpus

    return build_reference_corpus()
