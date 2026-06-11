from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.query.schemas import Citation, QueryMetadata, QueryRequest, QueryResponse
from app.api.v1.threads.models import Message, Thread
from app.core.constants import THREAD_TITLE_MAX_CHARS
from app.core.tracing import flush_tracer, traced_span
from app.graph.builder import stream_invoke_agent_graph
from app.graph.conversation import ChatTurn, load_recent_thread_messages
from app.graph.pipeline import run_agent_pipeline
from app.graph.state import GraphState
from app.llm.retrieval_query import build_retrieval_query
from app.retrieval.qdrant_store import QdrantStore, get_qdrant_store

log = structlog.get_logger(__name__)


async def _ensure_thread(session: AsyncSession, body: QueryRequest) -> UUID:
    """Create or load a conversation thread for the query."""
    thread_id = body.thread_id
    if thread_id is None:
        thread = Thread(title=body.message[:THREAD_TITLE_MAX_CHARS])
        session.add(thread)
        await session.commit()
        await session.refresh(thread)
        return thread.id

    thread = await session.get(Thread, thread_id)
    if thread is None:
        thread = Thread(id=thread_id, title=body.message[:THREAD_TITLE_MAX_CHARS])
        session.add(thread)
        await session.commit()
    return thread_id


def response_from_graph_state(
    state: GraphState,
    *,
    thread_id: UUID,
    checkpointer: Any | None,
) -> QueryResponse:
    citations = [
        Citation(
            document_id=citation.get("document_id"),
            chunk_id=citation.get("chunk_id"),
            snippet=citation.get("snippet", ""),
            score=citation.get("score"),
        )
        for citation in state.get("citations", [])
    ]
    metadata = QueryMetadata(
        route=state.get("route", "unknown"),  # type: ignore[arg-type]
        abstained=bool(state.get("abstained")),
        nodes_visited=list(state.get("nodes_visited", [])),
        retrieval_scores=list(state.get("retrieval_scores", [])),
        graph_checkpoint_enabled=checkpointer is not None,
    )
    return QueryResponse(
        answer=state.get("answer", ""),
        thread_id=thread_id,
        citations=citations,
        metadata=metadata,
    )


def _turns_to_history(turns: list[ChatTurn]) -> list[dict[str, str]]:
    return [{"role": turn.role, "content": turn.content} for turn in turns]


async def prepare_query_run(
    body: QueryRequest,
    session: AsyncSession,
) -> tuple[UUID, QdrantStore, str, list[dict[str, str]]]:
    thread_id = await _ensure_thread(session, body)
    prior_turns = await load_recent_thread_messages(session, thread_id)
    retrieval_query = await build_retrieval_query(prior_turns, body.message)
    session.add(Message(thread_id=thread_id, role="user", content=body.message))
    await session.commit()
    chat_history = _turns_to_history(prior_turns + [ChatTurn(role="user", content=body.message)])
    return thread_id, get_qdrant_store(), retrieval_query, chat_history


async def persist_assistant_message(
    session: AsyncSession,
    thread_id: UUID,
    response: QueryResponse,
) -> None:
    session.add(
        Message(
            thread_id=thread_id,
            role="assistant",
            content=response.answer,
            metadata_={
                **response.metadata.model_dump(mode="json"),
                "citations": [
                    citation.model_dump(mode="json", by_alias=True)
                    for citation in response.citations
                ],
            },
        )
    )
    await session.commit()
    log.info(
        "query_complete",
        thread_id=str(thread_id),
        route=response.metadata.route,
        checkpoint=response.metadata.graph_checkpoint_enabled,
    )


async def stream_query_graph(
    body: QueryRequest,
    session: AsyncSession,
    *,
    checkpointer: Any | None = None,
    compiled_graph: Any | None = None,
) -> AsyncIterator[str | QueryResponse]:
    """Yield graph node names as they complete, then the final ``QueryResponse``."""
    thread_id, qdrant, retrieval_query, chat_history = await prepare_query_run(body, session)
    async with traced_span("rag_query_stream", question=body.message, thread_id=str(thread_id)):
        async for item in stream_invoke_agent_graph(
            body.message,
            db=session,
            qdrant=qdrant,
            checkpointer=checkpointer,
            compiled_graph=compiled_graph,
            thread_id=str(thread_id),
            retrieval_query=retrieval_query,
            chat_history=chat_history,
        ):
            if isinstance(item, str):
                yield item
                continue
            response = response_from_graph_state(
                item,
                thread_id=thread_id,
                checkpointer=checkpointer,
            )
            await persist_assistant_message(session, thread_id, response)
            yield response
    flush_tracer()


async def run_query(
    body: QueryRequest,
    session: AsyncSession,
    *,
    checkpointer: Any | None = None,
    compiled_graph: Any | None = None,
) -> QueryResponse:
    """Persist messages, run LangGraph, and return a grounded response."""
    thread_id, qdrant, retrieval_query, chat_history = await prepare_query_run(body, session)
    async with traced_span("rag_query", question=body.message, thread_id=str(thread_id)):
        state = await run_agent_pipeline(
            body.message,
            session,
            qdrant,
            checkpointer=checkpointer,
            compiled_graph=compiled_graph,
            thread_id=str(thread_id),
            retrieval_query=retrieval_query,
            chat_history=chat_history,
        )
    flush_tracer()
    response = response_from_graph_state(
        state,
        thread_id=thread_id,
        checkpointer=checkpointer,
    )
    await persist_assistant_message(session, thread_id, response)
    return response
