"""Live pipeline progress events.

``emit_stage`` publishes a :class:`StageEvent` to LangGraph's custom stream.
When the graph runs through ``astream(stream_mode=["updates", "custom"])``
(the SSE path), these events surface to the client in real time. Outside a
graph run — unit tests, the non-streaming ``ainvoke`` path, the eval harness —
there is no stream writer in context and the call is a no-op, so domain code
can emit unconditionally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.config import get_stream_writer

from app.core.constants import STAGE_PHASE_START


@dataclass(frozen=True)
class StageEvent:
    """One pipeline progress signal (stage started / stage completed)."""

    stage: str
    phase: str = STAGE_PHASE_START
    detail: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """Shape used for the SSE ``status`` event body."""
        payload: dict[str, Any] = {"stage": self.stage, "phase": self.phase}
        if self.detail:
            payload["detail"] = self.detail
        return payload


def emit_stage(stage: str, *, phase: str = STAGE_PHASE_START, **detail: Any) -> None:
    """Emit a pipeline stage event to the active graph's custom stream, if any."""
    try:
        writer = get_stream_writer()
    except Exception:
        return
    try:
        writer(StageEvent(stage=stage, phase=phase, detail=detail or None))
    except Exception:
        # Never let progress reporting break the pipeline itself.
        return
