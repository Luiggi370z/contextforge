"""Application startup and shutdown lifecycle."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.graph.builder import build_agent_graph
from app.graph.checkpointer import postgres_checkpointer

log = structlog.get_logger(__name__)


@asynccontextmanager
async def application_lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Configure logging, LangGraph checkpointer, and shared app state.

    Example:
        Used as ``lifespan=application_lifespan`` on the FastAPI constructor.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info("application_starting", app_env=settings.app_env)
    log_kwargs = {"llm_provider": settings.llm_provider}
    if settings.llm_provider == "ollama":
        log_kwargs["ollama_base_url"] = settings.ollama_base_url
        log_kwargs["ollama_model"] = settings.ollama_model
    log.info("llm_provider_configured", **log_kwargs)

    async with postgres_checkpointer() as checkpointer:
        application.state.checkpointer = checkpointer
        application.state.agent_graph = build_agent_graph(checkpointer=checkpointer)
        if checkpointer is not None:
            log.info("langgraph_checkpointer_attached")
        else:
            log.warning("langgraph_checkpointer_unavailable")
        log.info("langgraph_agent_compiled", with_checkpointer=checkpointer is not None)

        try:
            application.state.arq_pool = await create_pool(
                RedisSettings.from_dsn(settings.redis_url)
            )
            log.info("arq_pool_connected", redis_url=settings.redis_url)
        except Exception as error:
            application.state.arq_pool = None
            log.warning("arq_pool_unavailable", error=str(error))

        yield

        arq_pool = getattr(application.state, "arq_pool", None)
        if arq_pool is not None:
            await arq_pool.close()
            log.info("arq_pool_closed")

    log.info("application_shutdown")
