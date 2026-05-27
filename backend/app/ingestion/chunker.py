from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.constants import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)
