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
SYSTEM_PROMPT_RETRIEVAL_QUERY = (
    "Rewrite the conversation into one standalone search query for retrieving policy documents. "
    "The query must answer what the user wants to know from their latest message. "
    "If the latest message is a follow-up, carry forward the subject being discussed. "
    "If the latest message changes topic, use only the new subject—do not keep keywords from the old topic. "
    "Reply with the search query text only—no quotes, labels, or explanation."
)
SYSTEM_PROMPT_GENERATE = (
    "Answer the user's question using only the provided context snippets. "
    "Be direct and concise. Quote or paraphrase the policy when it answers the question. "
    "If the policy applies only to a specific scope (for example production systems), "
    "state that scope clearly instead of claiming the documents are silent. "
    "Do not contradict a snippet that already answers part of the question. "
    "Only say context is insufficient when no snippet is relevant."
)


def format_numbered_contexts(contexts: Sequence[str]) -> str:
    return "\n\n".join(f"[{index + 1}] {value}" for index, value in enumerate(contexts))


def build_route_user_prompt(message: str) -> str:
    return message


def build_retrieval_query_user_prompt(*, conversation: str, latest_message: str) -> str:
    return f"Conversation:\n{conversation}\n\nLatest user message:\n{latest_message}"


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
    conversation: str | None = None,
) -> str:
    parts: list[str] = [f"Route: {route}"]
    if conversation:
        parts.extend(["", f"Conversation so far:\n{conversation}"])
    parts.extend(
        [
            "",
            f"Current question:\n{query}",
            "",
            f"Contexts:\n{format_numbered_contexts(contexts)}",
        ]
    )
    return "\n".join(parts)
