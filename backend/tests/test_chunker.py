from app.core.constants import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS
from app.ingestion.chunker import split_text


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
