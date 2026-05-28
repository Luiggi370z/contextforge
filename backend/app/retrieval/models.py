"""Retrieval domain types."""

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    score: float
    # Max dense/sparse score before RRF; used for abstain grading (RRF scores are much smaller).
    relevance_score: float | None = None

    def grading_score(self) -> float:
        return self.relevance_score if self.relevance_score is not None else self.score
