"""Query execution (agent graph orchestration)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.query.schemas import QueryRequest, QueryResponse
from app.api.v1.query.streaming import format_sse_event, stream_error_event, stream_query_events
from app.core.constants import (
    ERROR_INTERNAL_SERVER,
    SSE_EVENT_STATUS,
    SSE_FLUSH_COMMENT,
    SSE_STAGE_STARTED,
)
from app.core.exceptions import AppException
from app.graph.progress import StageEvent
from app.graph.runner import run_query, stream_query_graph

log = structlog.get_logger(__name__)


class QueryService:
    """Runs agentic RAG queries and persists thread messages."""

    async def execute(
        self,
        body: QueryRequest,
        session: AsyncSession,
        *,
        checkpointer: Any | None = None,
        compiled_graph: Any | None = None,
    ) -> QueryResponse:
        """Run a full query through the LangGraph pipeline."""
        return await run_query(
            body,
            session,
            checkpointer=checkpointer,
            compiled_graph=compiled_graph,
        )

    async def stream_execute(
        self,
        body: QueryRequest,
        session: AsyncSession,
        *,
        checkpointer: Any | None = None,
        compiled_graph: Any | None = None,
    ) -> AsyncIterator[str]:
        """Run a query and yield SSE frames (status, tokens, done)."""
        yield format_sse_event(SSE_EVENT_STATUS, {"stage": SSE_STAGE_STARTED})
        yield SSE_FLUSH_COMMENT
        await asyncio.sleep(0)

        try:
            result: QueryResponse | None = None
            async for item in stream_query_graph(
                body,
                session,
                checkpointer=checkpointer,
                compiled_graph=compiled_graph,
            ):
                if isinstance(item, StageEvent):
                    yield format_sse_event(SSE_EVENT_STATUS, item.to_payload())
                    yield SSE_FLUSH_COMMENT
                    await asyncio.sleep(0)
                    continue
                if isinstance(item, str):
                    # Back-compat with patched/fake stream generators in tests.
                    yield format_sse_event(SSE_EVENT_STATUS, {"stage": item})
                    yield SSE_FLUSH_COMMENT
                    await asyncio.sleep(0)
                    continue
                result = item
        except AppException as error:
            async for frame in stream_error_event(error.detail, error.correlation_id):
                yield frame
            return
        except Exception as error:
            log.exception("stream_query_failed", error=str(error))
            async for frame in stream_error_event(ERROR_INTERNAL_SERVER):
                yield frame
            return

        if result is None:
            async for frame in stream_error_event(ERROR_INTERNAL_SERVER):
                yield frame
            return

        async for frame in stream_query_events(
            result,
            graph_stages=[],
            emit_started=False,
        ):
            yield frame
