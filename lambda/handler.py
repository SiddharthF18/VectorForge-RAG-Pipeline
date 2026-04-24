"""AWS Lambda entry point for serverless refresh.

Deployment notes:
    - Package `src/vector_pipeline` and this handler in a Lambda layer or zip.
    - Set environment variables: PINECONE_API_KEY, PINECONE_INDEX, etc.
    - Trigger via EventBridge (cron) for daily refresh.
    - Memory >= 1024 MB recommended (SentenceTransformers model load).

Event payload:
    {"source": "aws", "limit": 20}
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):  # noqa: ANN001
    from vector_pipeline.embed import embed_texts
    from vector_pipeline.ingest import crawl_docs
    from vector_pipeline.preprocess import chunk_documents
    from vector_pipeline.store import upsert_chunks

    source = (event or {}).get("source", "aws")
    limit = int((event or {}).get("limit", 20))
    logger.info("Refreshing source=%s limit=%d", source, limit)

    docs = list(crawl_docs(source, max_pages=limit))
    chunks = list(chunk_documents(docs))
    if not chunks:
        return {"statusCode": 200, "body": json.dumps({"upserted": 0})}

    vectors = embed_texts(c.text for c in chunks)
    total = upsert_chunks(chunks, vectors)
    return {
        "statusCode": 200,
        "body": json.dumps({"source": source, "docs": len(docs), "upserted": total}),
    }
