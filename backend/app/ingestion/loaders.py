"""Route uploaded bytes to a text representation by file type.

A ``DocumentLoader`` turns raw bytes into a list of ``LoadedBlock`` (text +
page + section). Markdown keeps heading sections; PDF keeps page numbers; plain
text is one ``Body`` block. The chunker then splits each block's text while
carrying the block's page/section into chunk metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

_DEFAULT_SECTION = "Body"


@dataclass(frozen=True)
class LoadedBlock:
    text: str
    section: str = _DEFAULT_SECTION
    page: int | None = None


@runtime_checkable
class DocumentLoader(Protocol):
    def load(self, raw_bytes: bytes, *, filename: str) -> list[LoadedBlock]: ...


class TextLoader:
    """Plain text → a single Body block."""

    def load(self, raw_bytes: bytes, *, filename: str) -> list[LoadedBlock]:
        text = raw_bytes.decode("utf-8", errors="replace").strip()
        return [LoadedBlock(text=text, section=_DEFAULT_SECTION)] if text else []


class MarkdownLoader:
    """Markdown → one block per heading section (page=None)."""

    def load(self, raw_bytes: bytes, *, filename: str) -> list[LoadedBlock]:
        import re

        text = raw_bytes.decode("utf-8", errors="replace")
        heading = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
        matches = list(heading.finditer(text))
        if not matches:
            stripped = text.strip()
            return [LoadedBlock(text=stripped, section=_DEFAULT_SECTION)] if stripped else []
        blocks: list[LoadedBlock] = []
        head = text[: matches[0].start()].strip()
        if head:
            blocks.append(LoadedBlock(text=head, section=_DEFAULT_SECTION))
        for index, match in enumerate(matches):
            title = match.group(2).strip()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                blocks.append(LoadedBlock(text=body, section=title))
        return blocks


class PdfLoader:
    """PDF → one block per page (section='Page N'), via PyMuPDF."""

    def load(self, raw_bytes: bytes, *, filename: str) -> list[LoadedBlock]:
        import fitz  # PyMuPDF

        blocks: list[LoadedBlock] = []
        with fitz.open(stream=raw_bytes, filetype="pdf") as doc:
            for page_index in range(doc.page_count):
                page_number = page_index + 1
                text = str(doc.load_page(page_index).get_text("text")).strip()
                if text:
                    blocks.append(
                        LoadedBlock(text=text, section=f"Page {page_number}", page=page_number)
                    )
        return blocks


def _loader_for(filename: str, content_type: str | None) -> DocumentLoader:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".pdf" or (content_type or "").endswith("pdf"):
        return PdfLoader()
    if suffix in {".md", ".markdown"} or (content_type or "") == "text/markdown":
        return MarkdownLoader()
    return TextLoader()


def load_document(
    *, filename: str, raw_bytes: bytes, content_type: str | None
) -> list[LoadedBlock]:
    """Public entry: pick a loader by extension/content-type and load blocks."""
    return _loader_for(filename, content_type).load(raw_bytes, filename=filename)
