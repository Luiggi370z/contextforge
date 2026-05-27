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
