from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    database_url: str = (
        "postgresql+asyncpg://contextforge:contextforge@localhost:5433/contextforge"
    )
    # TODO(retrieval-backend): Branch ingest/retrieve on this when Postgres profile ships.
    # Values: qdrant (default) | postgres (pgvector dense + tsvector FTS).
    retrieval_backend: str = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "contextforge"

    llm_provider: str = "heuristic"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_backend: str = "sentence-transformers"
    cors_origins: str = "http://localhost:5173"

    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    rerank_backend: str = "lexical"
    grade_min_score: float = 0.25

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in ("production", "prod")


@lru_cache
def get_settings() -> Settings:
    return Settings()
