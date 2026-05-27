"""Route classification via Pydantic AI (top-level third-party imports)."""

from __future__ import annotations

from pydantic_ai import Agent

from app.core.config import get_settings
from app.llm.models import RouteDecision


async def decide_route(message: str) -> RouteDecision:
    settings = get_settings()
    agent = Agent(
        f"openai:{settings.openai_model}",
        output_type=RouteDecision,
        system_prompt=(
            "Classify the user query route: direct (no docs), single_hop_rag, or multi_hop."
        ),
    )
    result = await agent.run(message)
    return result.output
