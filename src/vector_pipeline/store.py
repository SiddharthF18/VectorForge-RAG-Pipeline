"""Pinecone vector store wrapper."""
from __future__ import annotations

import logging
import time
from typing import Iterable, Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings
from .preprocess import Chunk

logger = logging.getLogger(__name__)


def _client():
    from pinecone import Pinecone

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index() -> None:
    """Create the configured index on the free tier if it doesn't exist."""
    from pinecone import ServerlessSpec

    pc = _client()
    existing = {idx["name"] for idx in pc.list_indexes()}
    if settings.pinecone_index in existing:
        return
    logger.info("Creating Pinecone index '%s'", settings.pinecone_index)
    pc.create_index(
        name=settings.pinecone_index,
        dimension=settings.embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=settings.pinecone_cloud, region=settings.pinecone_region
        ),
    )
    # wait until ready
    for _ in range(30):
        desc = pc.describe_index(settings.pinecone_index)
        if desc.status.get("ready"):
            return
        time.sleep(2)


def _index():
    return _client().Index(settings.pinecone_index)


def _batch(items: Sequence, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=10))
def _upsert_batch(index, vectors: list[dict]) -> None:
    index.upsert(vectors=vectors)


def upsert_chunks(chunks: Iterable[Chunk], embeddings: Iterable[list[float]]) -> int:
    """Upsert (chunk, embedding) pairs to Pinecone. Returns total upserted."""
    ensure_index()
    index = _index()
    chunk_list = list(chunks)
    emb_list = list(embeddings)
    if len(chunk_list) != len(emb_list):
        raise ValueError("chunks and embeddings length mismatch")

    payload = [
        {
            "id": c.id,
            "values": v,
            "metadata": {
                "text": c.text[:2000],
                "source": c.source,
                "url": c.url,
                "title": c.title,
                "doc_id": c.doc_id,
                "position": c.position,
                "ingested_at": time.time(),
            },
        }
        for c, v in zip(chunk_list, emb_list)
    ]

    total = 0
    for batch in _batch(payload, settings.upsert_batch_size):
        _upsert_batch(index, batch)
        total += len(batch)
        logger.info("Upserted %d / %d chunks", total, len(payload))
    return total


def query(text_vector: list[float], top_k: int = 5, source: str | None = None) -> list[dict]:
    index = _index()
    flt = {"source": {"$eq": source}} if source else None
    res = index.query(
        vector=text_vector,
        top_k=top_k,
        include_metadata=True,
        filter=flt,
    )
    matches = []
    for m in res.get("matches", []):
        meta = m.get("metadata", {}) or {}
        matches.append(
            {
                "id": m.get("id"),
                "score": m.get("score"),
                "text": meta.get("text", ""),
                "source": meta.get("source", ""),
                "url": meta.get("url", ""),
                "title": meta.get("title", ""),
            }
        )
    return matches
