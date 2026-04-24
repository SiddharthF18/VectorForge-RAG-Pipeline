# AI-Ready Vector Ingestion Pipeline

End-to-end blueprint that ingests unstructured tech documentation, chunks and
embeds it, stores vectors in Pinecone, and serves retrieval through a FastAPI
endpoint plus a Streamlit chatbot UI. Includes Apache Spark for parallel
embedding at scale, an Apache Airflow DAG for daily refresh, and an AWS Lambda
handler for serverless re-ingestion.

```
Data Source -> Ingestion -> Preprocessing -> Chunking -> Embeddings -> Pinecone -> Retrieval API -> UI
```

## Layout

```
vector-pipeline/
  src/vector_pipeline/      Reusable library (ingest, preprocess, chunk, embed, store, retrieve)
  api/                      FastAPI retrieval service
  ui/                       Streamlit chatbot UI
  spark/                    PySpark job for parallel embedding (100k+ docs)
  airflow/dags/             Airflow DAG for daily refresh
  lambda/                   AWS Lambda handler for serverless refresh
  data/raw/                 Local cache of crawled / downloaded sources
  tests/                    Smoke tests
  config.example.env        Copy to .env and fill in
  requirements.txt          Python dependencies
```

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.env .env   # fill PINECONE_API_KEY etc.

# 1. Ingest -> chunk -> embed -> upsert to Pinecone
python -m vector_pipeline.run_ingest --source aws --limit 50

# 2. Start retrieval API
uvicorn api.main:app --reload --port 8000

# 3. Start chatbot UI
streamlit run ui/app.py
```

## Datasets

Default source is **tech documentation** (AWS, Spark, Kubernetes). Switch via
`--source {aws,spark,k8s,arxiv,pdf}`. arXiv and PDF ingestion are also wired.

## Scaling

The PySpark job in `spark/embed_job.py` parallelises chunking + embedding over
the local Spark master and writes results in batches to Pinecone. Tested as a
shape against 100k+ synthetic chunks.

## Automation

- `airflow/dags/refresh_dag.py` — daily DAG: ingest -> chunk -> embed -> upsert.
- `lambda/handler.py` — invoke via EventBridge (cron) for serverless refresh of
  a single source.

## Configuration

All knobs live in `src/vector_pipeline/config.py` and the `.env` file. Defaults
target Pinecone's free tier and the `all-MiniLM-L6-v2` SentenceTransformer
(384-dim).
