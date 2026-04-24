"""High-level retrieval helper combining embedding + Pinecone query."""
from __future__ import annotations

from .embed import embed_query
from .store import query as vector_query


def retrieve(query_text: str, top_k: int = 5, source: str | None = None) -> list[dict]:
    vec = embed_query(query_text)
    return vector_query(vec, top_k=top_k, source=source)
