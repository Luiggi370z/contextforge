"""Route classification via Instructor + OpenAI (top-level third-party imports)."""

from __future__ import annotations

import instructor
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.llm.models import RouteDecision


async def decide_route(message: str) -> RouteDecision:
    settings = get_settings()
    client = instructor.from_openai(AsyncOpenAI(api_key=settings.openai_api_key))
    return await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user query route: direct (no docs), "
                    "single_hop_rag, or multi_hop."
                ),
            },
            {"role": "user", "content": message},
        ],
        response_model=RouteDecision,
    )
