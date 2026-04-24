"""Text cleaning and token-aware chunking with overlap."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator

from .config import settings
from .ingest import Document, _doc_id

_WS_RE = re.compile(r"\s+")
_BOILERPLATE = (
    "cookie policy",
    "all rights reserved",
    "terms of service",
    "privacy policy",
)


def clean_text(text: str) -> str:
    text = _WS_RE.sub(" ", text).strip()
    lowered = text.lower()
    for marker in _BOILERPLATE:
        idx = lowered.find(marker)
        if idx > 0:
            text = text[:idx].strip()
            lowered = text.lower()
    return text


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    source: str
    url: str
    title: str
    position: int


def _approx_tokens(text: str) -> list[str]:
    """Whitespace splitter that approximates BPE tokens (~1.3 words/token)."""
    return text.split()


def chunk_text(
    text: str,
    chunk_tokens: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    chunk_tokens = chunk_tokens or settings.chunk_tokens
    overlap = overlap if overlap is not None else settings.chunk_overlap
    if overlap >= chunk_tokens:
        raise ValueError("overlap must be smaller than chunk size")

    words = _approx_tokens(text)
    if not words:
        return []
    # ~1 token per 0.75 words is the common heuristic; we use words directly
    # because all-MiniLM-L6-v2 has a 256-token cap and 400 words ~= 300 tokens.
    step = chunk_tokens - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_tokens]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + chunk_tokens >= len(words):
            break
    return chunks


def chunk_documents(docs: Iterable[Document]) -> Iterator[Chunk]:
    for doc in docs:
        cleaned = clean_text(doc.text)
        if not cleaned:
            continue
        for i, piece in enumerate(chunk_text(cleaned)):
            yield Chunk(
                id=_doc_id(doc.source, f"{doc.url}#{i}"),
                doc_id=doc.id,
                text=piece,
                source=doc.source,
                url=doc.url,
                title=doc.title,
                position=i,
            )
