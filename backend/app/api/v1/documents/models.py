"""Domain types for the documents entity (non-Pydantic)."""

from __future__ import annotations

from dataclasses import dataclass

from app.db.models import Document


@dataclass(frozen=True)
class DocumentListResult:
    """Documents returned from a list operation."""

    items: list[Document]
    total: int
