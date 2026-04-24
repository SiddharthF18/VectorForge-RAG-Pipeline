"""CLI entry-point: ingest -> chunk -> embed -> upsert."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import settings
from .embed import embed_texts
from .ingest import Document, crawl_docs, fetch_arxiv, load_pdf
from .preprocess import chunk_documents
from .store import upsert_chunks

logger = logging.getLogger("vector_pipeline.run_ingest")


def _gather(source: str, limit: int, query: str | None, pdf_path: str | None):
    if source in {"aws", "spark", "k8s"}:
        yield from crawl_docs(source, max_pages=limit)
    elif source == "arxiv":
        yield from fetch_arxiv(query or "large language models", max_results=limit)
    elif source == "pdf":
        if not pdf_path:
            raise SystemExit("--pdf is required when --source pdf")
        for p in Path(pdf_path).rglob("*.pdf"):
            yield load_pdf(p)
    else:
        raise SystemExit(f"Unknown source: {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone")
    parser.add_argument(
        "--source", default="aws", choices=["aws", "spark", "k8s", "arxiv", "pdf"]
    )
    parser.add_argument("--limit", type=int, default=25, help="max pages / results")
    parser.add_argument("--query", default=None, help="arxiv search query")
    parser.add_argument("--pdf", default=None, help="path to a directory of PDFs")
    args = parser.parse_args()

    docs: list[Document] = list(_gather(args.source, args.limit, args.query, args.pdf))
    logger.info("Fetched %d documents from %s", len(docs), args.source)

    chunks = list(chunk_documents(docs))
    logger.info("Produced %d chunks (avg %d chars)", len(chunks),
                (sum(len(c.text) for c in chunks) // max(len(chunks), 1)))

    if not chunks:
        logger.warning("Nothing to embed; exiting")
        return

    vectors = embed_texts((c.text for c in chunks))
    logger.info("Embedded %d chunks with %s", len(vectors), settings.embedding_model)

    total = upsert_chunks(chunks, vectors)
    logger.info("Done. Upserted %d vectors to index '%s'", total, settings.pinecone_index)


if __name__ == "__main__":
    main()
