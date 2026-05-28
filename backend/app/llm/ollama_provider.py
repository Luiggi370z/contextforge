"""Ollama provider for structured and text LLM tasks."""

from __future__ import annotations

import json

import httpx

from app.llm.models import AnswerValidation, RetrievalGrade, RouteDecision, RouteKind
from app.graph.conversation import ChatTurn, format_chat_history
from app.llm.prompts.ollama import (
    SYSTEM_PROMPT_GENERATE,
    SYSTEM_PROMPT_GRADE,
    SYSTEM_PROMPT_RETRIEVAL_QUERY,
    SYSTEM_PROMPT_ROUTE,
    SYSTEM_PROMPT_VALIDATE,
    build_generate_user_prompt,
    build_grade_user_prompt,
    build_retrieval_query_user_prompt,
    build_route_user_prompt,
    build_validate_user_prompt,
)
from app.retrieval.models import RetrievedChunk

OLLAMA_CHAT_ENDPOINT = "/api/chat"
OLLAMA_TIMEOUT_SECONDS = 30.0


def _build_chat_payload(*, model: str, system_prompt: str, user_prompt: str, as_json: bool) -> dict:
    payload: dict[str, object] = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if as_json:
        payload["format"] = "json"
    return payload


def _extract_message_content(response_json: dict) -> str:
    message = response_json.get("message", {})
    if not isinstance(message, dict):
        raise ValueError("invalid_ollama_response_message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("invalid_ollama_response_content")
    return content.strip()


def _parse_json_content(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("invalid_ollama_json_payload")
    return parsed


async def _chat_async(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    as_json: bool,
) -> str:
    payload = _build_chat_payload(
        model=model, system_prompt=system_prompt, user_prompt=user_prompt, as_json=as_json
    )
    async with httpx.AsyncClient(base_url=base_url, timeout=OLLAMA_TIMEOUT_SECONDS) as client:
        response = await client.post(OLLAMA_CHAT_ENDPOINT, json=payload)
        response.raise_for_status()
    return _extract_message_content(response.json())


def _chat_sync(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    as_json: bool,
) -> str:
    payload = _build_chat_payload(
        model=model, system_prompt=system_prompt, user_prompt=user_prompt, as_json=as_json
    )
    with httpx.Client(base_url=base_url, timeout=OLLAMA_TIMEOUT_SECONDS) as client:
        response = client.post(OLLAMA_CHAT_ENDPOINT, json=payload)
        response.raise_for_status()
    return _extract_message_content(response.json())


async def decide_route(
    message: str,
    *,
    base_url: str,
    model: str,
) -> RouteDecision:
    content = await _chat_async(
        base_url=base_url,
        model=model,
        system_prompt=SYSTEM_PROMPT_ROUTE,
        user_prompt=build_route_user_prompt(message),
        as_json=True,
    )
    return RouteDecision.model_validate(_parse_json_content(content))


def grade_retrieval(
    *,
    query: str,
    chunks: list[RetrievedChunk],
    threshold: float,
    base_url: str,
    model: str,
) -> RetrievalGrade:
    user_prompt = build_grade_user_prompt(
        query=query,
        threshold=threshold,
        contexts=[chunk.content for chunk in chunks],
    )
    content = _chat_sync(
        base_url=base_url,
        model=model,
        system_prompt=SYSTEM_PROMPT_GRADE,
        user_prompt=user_prompt,
        as_json=True,
    )
    return RetrievalGrade.model_validate(_parse_json_content(content))


def validate_answer(
    *,
    answer: str,
    contexts: list[str],
    base_url: str,
    model: str,
) -> AnswerValidation:
    user_prompt = build_validate_user_prompt(answer=answer, contexts=contexts)
    content = _chat_sync(
        base_url=base_url,
        model=model,
        system_prompt=SYSTEM_PROMPT_VALIDATE,
        user_prompt=user_prompt,
        as_json=True,
    )
    return AnswerValidation.model_validate(_parse_json_content(content))


async def condense_retrieval_query(
    *,
    conversation: str,
    latest_message: str,
    base_url: str,
    model: str,
) -> str:
    """Rewrite multi-turn chat into one search query for hybrid retrieval."""
    user_prompt = build_retrieval_query_user_prompt(
        conversation=conversation,
        latest_message=latest_message,
    )
    return await _chat_async(
        base_url=base_url,
        model=model,
        system_prompt=SYSTEM_PROMPT_RETRIEVAL_QUERY,
        user_prompt=user_prompt,
        as_json=False,
    )


def generate_from_context(
    *,
    query: str,
    contexts: list[str],
    route: RouteKind,
    base_url: str,
    model: str,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    conversation: str | None = None
    if chat_history and len(chat_history) > 1:
        prior = [
            ChatTurn(role=turn["role"], content=turn["content"])
            for turn in chat_history[:-1]
        ]
        conversation = format_chat_history(prior)
    user_prompt = build_generate_user_prompt(
        route=route,
        query=query,
        contexts=contexts,
        conversation=conversation,
    )
    return _chat_sync(
        base_url=base_url,
        model=model,
        system_prompt=SYSTEM_PROMPT_GENERATE,
        user_prompt=user_prompt,
        as_json=False,
    )
