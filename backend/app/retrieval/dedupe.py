"""Deduplicate retrieved chunks and API citations."""

from __future__ import annotations

from app.retrieval.models import RetrievedChunk


def dedupe_chunks_by_content(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep one chunk per identical body (first = highest rank after rerank)."""
    seen: set[str] = set()
    unique: list[RetrievedChunk] = []
    for chunk in chunks:
        key = chunk.content.strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def dedupe_citations(citations: list[dict]) -> list[dict]:
    """Drop duplicate citations by chunk id or identical snippet text."""
    seen_chunk_ids: set[str] = set()
    seen_snippets: set[str] = set()
    unique: list[dict] = []
    for citation in citations:
        chunk_id = str(citation.get("chunk_id") or "")
        snippet = str(citation.get("snippet") or "").strip()
        if chunk_id and chunk_id in seen_chunk_ids:
            continue
        if snippet and snippet in seen_snippets:
            continue
        if chunk_id:
            seen_chunk_ids.add(chunk_id)
        if snippet:
            seen_snippets.add(snippet)
        unique.append(citation)
    return unique
