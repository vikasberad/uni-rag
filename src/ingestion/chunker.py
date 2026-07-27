"""Chunk documents for embedding. Uses LangChain's RecursiveCharacterTextSplitter
with a plain-Python fallback so the module works even without langchain installed."""
from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.loader import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _HAS_LC = True
except ImportError:
    _HAS_LC = False


@dataclass
class Chunk:
    chunk_id: str
    applicant_id: str
    doc_type: str
    text: str
    metadata: dict


def _fallback_split(text: str, size: int, overlap: int) -> list[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def chunk_documents(docs: list[Document], chunk_size: int = 700,
                    chunk_overlap: int = 100) -> list[Chunk]:
    if _HAS_LC:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "])
        split = splitter.split_text
    else:
        split = lambda t: _fallback_split(t, chunk_size, chunk_overlap)  # noqa: E731

    chunks: list[Chunk] = []
    for doc in docs:
        for j, piece in enumerate(split(doc.text)):
            chunks.append(Chunk(
                chunk_id=f"{doc.applicant_id}:{doc.doc_type}:{j}",
                applicant_id=doc.applicant_id,
                doc_type=doc.doc_type,
                text=piece.strip(),
                metadata=doc.metadata,
            ))
    return chunks
