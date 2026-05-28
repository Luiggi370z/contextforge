"""Ollama system and user prompts."""

from __future__ import annotations

from collections.abc import Sequence

from app.llm.models import RouteKind

SYSTEM_PROMPT_ROUTE = (
    "Classify the user query route as one of: direct, single_hop_rag, multi_hop. "
    "Return JSON only with fields route, confidence, and reasoning."
)
SYSTEM_PROMPT_GRADE = (
    "Grade whether retrieved context is relevant enough to answer the query. "
    "Return JSON only with fields relevant, score, and should_abstain."
)
SYSTEM_PROMPT_VALIDATE = (
    "Check whether the answer is grounded in the provided contexts. "
    "Return JSON only with fields grounded and issues."
)
SYSTEM_PROMPT_GENERATE = (
    "Answer using only the provided context snippets. "
    "If context is insufficient, respond exactly: "
    "'I do not have enough grounded context to answer that yet.'"
)


def format_numbered_contexts(contexts: Sequence[str]) -> str:
    return "\n\n".join(f"[{index + 1}] {value}" for index, value in enumerate(contexts))


def build_route_user_prompt(message: str) -> str:
    return message


def build_grade_user_prompt(
    *,
    query: str,
    threshold: float,
    contexts: Sequence[str],
) -> str:
    return (
        f"Query:\n{query}\n\n"
        f"Abstain threshold score:\n{threshold}\n\n"
        f"Retrieved chunks:\n{format_numbered_contexts(contexts)}"
    )


def build_validate_user_prompt(*, answer: str, contexts: Sequence[str]) -> str:
    return f"Answer:\n{answer}\n\nContexts:\n{format_numbered_contexts(contexts)}"


def build_generate_user_prompt(
    *,
    route: RouteKind,
    query: str,
    contexts: Sequence[str],
) -> str:
    return (
        f"Route: {route}\n\n"
        f"Query:\n{query}\n\n"
        f"Contexts:\n{format_numbered_contexts(contexts)}"
    )
