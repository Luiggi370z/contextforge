from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import DEFAULT_MAX_UPLOAD_BYTES

LLMProvider = Literal["heuristic", "openai", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    database_url: str = (
        "postgresql+asyncpg://contextforge:contextforge@localhost:5434/contextforge"
    )
    redis_url: str = "redis://localhost:6379"
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    contextual_retrieval_enabled: bool = False
    # TODO(retrieval-backend): Branch ingest/retrieve on this when Postgres profile ships.
    # Values: qdrant (default) | postgres (pgvector dense + tsvector FTS).
    retrieval_backend: str = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "contextforge"

    llm_provider: LLMProvider = "heuristic"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_backend: str = "sentence-transformers"
    # 384 for all-MiniLM-L6-v2; 1024 for BGE-M3. Must match the active embedder.
    embedding_dim: int = 384
    sparse_embedding_model: str = "Qdrant/bm25"
    # Named-vector keys inside the hybrid Qdrant collection.
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    cors_origins: str = "http://localhost:5173"

    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    rerank_backend: str = "lexical"
    # Max seconds the cross-encoder reranker may run before we fall back to the
    # RRF-ordered candidate list. Keeps the streamed read-path responsive when
    # the model is cold/slow. 0 or negative disables the timeout.
    rerank_timeout_s: float = 5.0
    # Lexical rerank / dense relevance scores are typically 0–1.
    grade_min_score: float = 0.25
    # Cross-encoder logits (ms-marco); negative scores are often irrelevant.
    grade_min_score_cross_encoder: float = 0.0

    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in ("production", "prod")


@lru_cache
def get_settings() -> Settings:
    return Settings()
