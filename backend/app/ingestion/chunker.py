"""Structure-aware chunking with Anthropic-style contextual prefixes.

The splitter walks the document by Markdown headings, then falls back to a
recursive character splitter inside each section. Every emitted chunk carries:

- ``body``: the raw chunk text the user will see in citations.
- ``context_prefix``: ``Document: {filename} > Section: {section}`` — prepended
  to ``body`` before embedding so dense retrieval picks up the section topic
  even when the body itself does not repeat it.
- ``content``: the concatenated string we actually embed.
- ``metadata``: ``{ "filename", "section", "chunk_index" }`` so the API can
  display the section a citation came from.

This is the pattern Anthropic showed in their contextual retrieval blog post
(2024-09): pre-pending lightweight structural context to each chunk improves
retrieval recall meaningfully without changing the retriever or model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.constants import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_DEFAULT_SECTION = "Body"


@dataclass(frozen=True)
class ChunkPiece:
    """One unit of text plus the metadata we keep for retrieval and citations."""

    body: str
    context_prefix: str
    metadata: dict

    @property
    def content(self) -> str:
        """What we hand to the embedder and the BM25 index."""
        if not self.context_prefix:
            return self.body
        return f"{self.context_prefix}\n\n{self.body}"


def _iter_sections(text: str) -> list[tuple[str, str]]:
    """Yield (section_title, section_body) for each Markdown heading-delimited block."""
    matches = list(_HEADING_PATTERN.finditer(text))
    if not matches:
        return [(_DEFAULT_SECTION, text.strip())]
    sections: list[tuple[str, str]] = []
    head = text[: matches[0].start()].strip()
    if head:
        sections.append((_DEFAULT_SECTION, head))
    for index, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def split_text_into_chunks(
    text: str,
    *,
    filename: str = "document",
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[ChunkPiece]:
    """Split ``text`` into chunks that carry their section-level context.

    Example:
        >>> pieces = split_text_into_chunks("# PTO\\n20 days", filename="pto.md")
        >>> pieces[0].body
        '20 days'
        >>> pieces[0].context_prefix
        'Document: pto.md > Section: PTO'
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    pieces: list[ChunkPiece] = []
    chunk_index = 0
    for section_title, section_body in _iter_sections(text):
        context_prefix = f"Document: {filename} > Section: {section_title}"
        for body in splitter.split_text(section_body):
            if not body.strip():
                continue
            pieces.append(
                ChunkPiece(
                    body=body,
                    context_prefix=context_prefix,
                    metadata={
                        "filename": filename,
                        "section": section_title,
                        "chunk_index": chunk_index,
                    },
                )
            )
            chunk_index += 1
    return pieces


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Back-compat helper used by tests; returns just the body strings.

    New code should call :func:`split_text_into_chunks` so the section/file context
    is preserved for retrieval.
    """
    return [
        piece.body
        for piece in split_text_into_chunks(
            text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    ]
