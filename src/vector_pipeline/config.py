"""Central configuration loaded from environment variables."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional at runtime
    pass


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "tech-docs")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))

    chunk_tokens: int = int(os.getenv("CHUNK_TOKENS", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "60"))

    upsert_batch_size: int = int(os.getenv("UPSERT_BATCH_SIZE", "100"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    data_dir: Path = Path(os.getenv("DATA_DIR", "data/raw"))


settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("vector_pipeline")
