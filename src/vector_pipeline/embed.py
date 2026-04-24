"""Embedding generation using SentenceTransformers."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable

from .config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model lazily and cache it across calls."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def embed_texts(
    texts: Iterable[str],
    batch_size: int = 64,
    normalize: bool = True,
) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
