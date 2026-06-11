"""Thread history helpers for multi-turn RAG."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.threads.models import Message
from app.core.constants import CONVERSATION_HISTORY_LIMIT


@dataclass(frozen=True)
class ChatTurn:
    role: str
    content: str


async def load_recent_thread_messages(
    session: AsyncSession,
    thread_id: UUID,
    *,
    limit: int = CONVERSATION_HISTORY_LIMIT,
) -> list[ChatTurn]:
    """Return recent messages for a thread in chronological order."""
    result = await session.execute(
        select(Message)
        .where(Message.thread_id == thread_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    rows = list(result.scalars().all())
    rows.reverse()
    return [ChatTurn(role=row.role, content=row.content) for row in rows]


def format_chat_history(turns: list[ChatTurn]) -> str:
    """Format prior turns for LLM prompts."""
    lines: list[str] = []
    for turn in turns:
        role = turn.role.capitalize()
        lines.append(f"{role}: {turn.content.strip()}")
    return "\n\n".join(lines)
