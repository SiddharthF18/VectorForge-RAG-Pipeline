"""PySpark job: parallel chunking + embedding for 100k+ documents.

Usage:
    spark-submit --master local[*] spark/embed_job.py \
        --input data/raw/*.txt --output data/embeddings.parquet

The job:
    1. Reads raw documents from a directory or glob.
    2. Chunks them in parallel.
    3. Embeds each partition with SentenceTransformers (model loaded once
       per executor via a broadcasted singleton).
    4. Writes (id, source, url, text, embedding) to Parquet, optionally
       upserting into Pinecone in driver-side batches.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pyspark.sql import Row, SparkSession  # noqa: E402

from vector_pipeline.config import settings  # noqa: E402
from vector_pipeline.preprocess import chunk_text, clean_text  # noqa: E402


def _embed_partition(rows):
    """Executor-side: load model once, embed an iterator of rows."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model)
    buffer, meta = [], []
    BATCH = 64
    for r in rows:
        buffer.append(r["text"])
        meta.append(r)
        if len(buffer) >= BATCH:
            vecs = model.encode(buffer, normalize_embeddings=True).tolist()
            for m, v in zip(meta, vecs):
                yield Row(
                    id=m["id"],
                    source=m["source"],
                    url=m["url"],
                    text=m["text"],
                    embedding=v,
                )
            buffer, meta = [], []
    if buffer:
        vecs = model.encode(buffer, normalize_embeddings=True).tolist()
        for m, v in zip(meta, vecs):
            yield Row(
                id=m["id"],
                source=m["source"],
                url=m["url"],
                text=m["text"],
                embedding=v,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="glob of raw .txt files")
    parser.add_argument("--output", required=True, help="parquet output path")
    parser.add_argument("--source", default="bulk")
    parser.add_argument("--upsert", action="store_true", help="also upsert to Pinecone")
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("vector-pipeline-embed")
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .getOrCreate()
    )
    sc = spark.sparkContext

    files_rdd = sc.wholeTextFiles(args.input)

    def _to_chunks(item):
        path, content = item
        cleaned = clean_text(content)
        for i, piece in enumerate(chunk_text(cleaned)):
            yield {
                "id": f"{Path(path).stem}-{i}",
                "source": args.source,
                "url": path,
                "text": piece,
            }

    chunks_rdd = files_rdd.flatMap(_to_chunks)
    embedded_rdd = chunks_rdd.mapPartitions(_embed_partition)

    df = spark.createDataFrame(embedded_rdd)
    df.write.mode("overwrite").parquet(args.output)
    print(f"Wrote {df.count()} embeddings to {args.output}")

    if args.upsert:
        # Driver-side upsert in batches to keep Pinecone client off executors.
        from vector_pipeline.store import ensure_index, _index

        ensure_index()
        index = _index()
        rows = df.collect()
        BATCH = settings.upsert_batch_size
        for i in range(0, len(rows), BATCH):
            batch = rows[i : i + BATCH]
            index.upsert(
                vectors=[
                    {
                        "id": r["id"],
                        "values": r["embedding"],
                        "metadata": {
                            "text": r["text"][:2000],
                            "source": r["source"],
                            "url": r["url"],
                        },
                    }
                    for r in batch
                ]
            )
        print(f"Upserted {len(rows)} vectors to Pinecone index {settings.pinecone_index}")

    spark.stop()


if __name__ == "__main__":
    main()
