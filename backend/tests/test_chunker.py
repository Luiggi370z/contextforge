from app.core.constants import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS
from app.ingestion.chunker import split_text, split_text_into_chunks


def test_split_text_returns_single_chunk_for_short_input():
    text = "Short policy note."
    chunks = split_text(text)
    assert chunks == [text]


def test_split_text_produces_multiple_chunks_for_long_input():
    paragraph = "word " * 200
    text = (paragraph + "\n\n") * 6
    chunks = split_text(text, chunk_size=CHUNK_SIZE_CHARS, chunk_overlap=CHUNK_OVERLAP_CHARS)
    assert len(chunks) > 1
    assert all(len(chunk) <= CHUNK_SIZE_CHARS + 50 for chunk in chunks)


def test_split_text_into_chunks_carries_section_context():
    text = (
        "# PTO Policy\n\n"
        "Employees accrue 20 PTO days per year.\n\n"
        "# Security Policy\n\n"
        "Access to production requires MFA."
    )
    pieces = split_text_into_chunks(text, filename="employee-handbook.md")
    sections = {piece.metadata["section"] for piece in pieces}
    assert sections == {"PTO Policy", "Security Policy"}
    pto = next(p for p in pieces if p.metadata["section"] == "PTO Policy")
    assert "Document: employee-handbook.md" in pto.context_prefix
    assert "Section: PTO Policy" in pto.context_prefix
    # The embedded content includes the section prefix; the body does not.
    assert pto.context_prefix in pto.content
    assert pto.body == "Employees accrue 20 PTO days per year."


def test_split_text_into_chunks_falls_back_when_no_headings():
    """Documents without markdown headings still chunk under a default section."""
    pieces = split_text_into_chunks("Plain text policy.", filename="raw.txt")
    assert len(pieces) == 1
    assert pieces[0].metadata["section"] == "Body"
    assert pieces[0].body == "Plain text policy."
