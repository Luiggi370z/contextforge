"""OpenAI prompts (mirror the Ollama prompt structure for cross-provider parity)."""

from __future__ import annotations

from collections.abc import Sequence

from app.llm.models import RouteKind

SYSTEM_PROMPT_ROUTE = (
    "Classify the user query into exactly one of: direct, single_hop_rag, multi_hop. "
    "direct = greetings or small talk that needs no documents. "
    "single_hop_rag = factual question answerable from one or two policy chunks. "
    "multi_hop = comparison or multi-part question that needs evidence from several chunks. "
    "Return a structured decision with a brief reasoning."
)
SYSTEM_PROMPT_GRADE = (
    "Decide whether the retrieved policy chunks are sufficient to answer the user's question. "
    "Set should_abstain=true only when no chunk meaningfully helps. "
    "Score is your confidence that the chunks are relevant (0..1)."
)
SYSTEM_PROMPT_VALIDATE = (
    "Decide whether the answer is entailed by the provided contexts. "
    "grounded=true means every factual claim in the answer is supported by at least one context. "
    "If the answer says it does not know, treat it as grounded. "
    "List any unsupported claims under issues."
)
SYSTEM_PROMPT_RETRIEVAL_QUERY = (
    "You rewrite a multi-turn conversation into one standalone search query for retrieving "
    "policy documents. The query must answer what the user wants to know from their latest "
    "message. If the latest message is a short follow-up, carry forward the subject from the "
    "prior turn. If the latest message changes topic, use only the new subject and drop the old "
    "keywords. references_prior_turn must be true when the latest message would be ambiguous "
    "without the prior turn (e.g. pronouns, 'is it', 'what about')."
)
SYSTEM_PROMPT_GENERATE = (
    "You answer the user's question using only the provided context snippets. "
    "Be direct and concise. Quote or paraphrase the policy when it answers the question. "
    "Cite the snippet you used at the end of each claim using [1], [2] tags matching the "
    "numbered context blocks. Every factual claim must be backed by at least one tag. "
    "If the policy applies only to a specific scope (e.g. production systems), state that scope "
    "clearly instead of claiming the documents are silent. "
    "Do not contradict a snippet that already answers part of the question. "
    "If no snippet is relevant, say: \"I don't have enough information in the indexed "
    "documents to answer that.\""
)


def _format_numbered_contexts(contexts: Sequence[str]) -> str:
    return "\n\n".join(f"[{index + 1}] {value}" for index, value in enumerate(contexts))


def build_retrieval_query_user_prompt(*, conversation: str, latest_message: str) -> str:
    return f"Conversation:\n{conversation}\n\nLatest user message:\n{latest_message}"


def build_grade_user_prompt(
    *,
    query: str,
    threshold: float,
    contexts: Sequence[str],
    conversation: str | None = None,
) -> str:
    parts: list[str] = []
    if conversation:
        parts.extend([f"Conversation so far:\n{conversation}", ""])
    parts.extend(
        [
            f"Question to answer:\n{query}",
            "",
            f"Reference score threshold (reranker): {threshold}",
            "",
            f"Retrieved chunks:\n{_format_numbered_contexts(contexts)}",
        ]
    )
    return "\n".join(parts)


def build_validate_user_prompt(*, answer: str, contexts: Sequence[str]) -> str:
    return f"Answer:\n{answer}\n\nContexts:\n{_format_numbered_contexts(contexts)}"


def build_generate_user_prompt(
    *,
    route: RouteKind,
    query: str,
    contexts: Sequence[str],
    conversation: str | None = None,
    retrieval_query: str | None = None,
) -> str:
    parts: list[str] = [f"Route: {route}"]
    if conversation:
        parts.extend(["", f"Conversation so far:\n{conversation}"])
    parts.extend(["", f"Current question:\n{query}"])
    if retrieval_query and retrieval_query.strip() and retrieval_query.strip() != query.strip():
        parts.extend(
            [
                "",
                f"Resolved subject (use this to disambiguate follow-ups):\n{retrieval_query}",
            ]
        )
    parts.extend(["", f"Contexts:\n{_format_numbered_contexts(contexts)}"])
    return "\n".join(parts)
