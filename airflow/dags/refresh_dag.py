"""Airflow DAG: daily refresh of the vector index.

Drop this file into your AIRFLOW_HOME/dags/ directory. The DAG runs the
ingest -> chunk -> embed -> upsert pipeline once per source per day.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "data-platform",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

SOURCES = ["aws", "spark", "k8s"]


def _run(source: str) -> None:
    # Imported inside the task so Airflow doesn't need the heavy deps at parse time.
    from vector_pipeline.embed import embed_texts
    from vector_pipeline.ingest import crawl_docs
    from vector_pipeline.preprocess import chunk_documents
    from vector_pipeline.store import upsert_chunks

    docs = list(crawl_docs(source, max_pages=50))
    chunks = list(chunk_documents(docs))
    if not chunks:
        return
    vectors = embed_texts(c.text for c in chunks)
    upsert_chunks(chunks, vectors)


with DAG(
    dag_id="vector_pipeline_daily_refresh",
    description="Daily refresh of tech-docs vector index",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule="0 3 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["rag", "pinecone"],
) as dag:
    for src in SOURCES:
        PythonOperator(
            task_id=f"refresh_{src}",
            python_callable=_run,
            op_kwargs={"source": src},
        )
