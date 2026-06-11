"""DocumentLoader routing: .md / .txt / .pdf produce LoadedBlocks with page/section."""

from __future__ import annotations

import pytest

from app.ingestion.loaders import LoadedBlock, load_document


def test_markdown_loader_preserves_headings():
    blocks = load_document(
        filename="pto.md",
        raw_bytes=b"# PTO\n\nFull-time staff accrue 20 PTO days.\n\n# Rollover\n\nUp to 10 days.",
        content_type="text/markdown",
    )
    assert all(isinstance(block, LoadedBlock) for block in blocks)
    sections = {block.section for block in blocks}
    assert "PTO" in sections
    assert "Rollover" in sections
    # markdown has no pages
    assert all(block.page is None for block in blocks)


def test_txt_loader_single_body_section():
    blocks = load_document(
        filename="notes.txt",
        raw_bytes=b"just some plain text without headings",
        content_type="text/plain",
    )
    assert len(blocks) >= 1
    assert blocks[0].section == "Body"
    assert "plain text" in blocks[0].text


def test_unknown_extension_falls_back_to_text():
    blocks = load_document(
        filename="mystery.dat",
        raw_bytes=b"hello world",
        content_type=None,
    )
    assert blocks
    assert "hello world" in blocks[0].text


def test_pdf_loader_extracts_page_text():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Full-time staff accrue 20 PTO days per calendar year.")
    pdf_bytes = doc.tobytes()
    doc.close()

    blocks = load_document(
        filename="policy.pdf", raw_bytes=pdf_bytes, content_type="application/pdf"
    )
    assert blocks
    assert blocks[0].page == 1
    assert "PTO days" in blocks[0].text
