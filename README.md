# 🔍 VectorForge — AI-Ready Vector Ingestion Pipeline for RAG Systems

> A production-grade pipeline that crawls technical documentation, embeds it using a local AI model, stores vectors in Pinecone, and serves a semantic search API + chatbot UI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-red?logo=streamlit)
![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📸 Demo

Ask any question about AWS, Apache Spark, or Kubernetes — the system finds the most semantically relevant documentation passages in milliseconds.

```
Query: "How does Apache Spark handle distributed data processing?"

Top 5 passages (2194 ms)
1. Apache Spark™ - Unified Engine — Apache Spark is a multi-language engine for
   executing data engineering, data science, and machine learning on single-node
   machines or clusters...
2. Examples | Apache Spark — Spark allows for efficient execution of the query
   because it parallelizes this computation...
```

---

## 🏗️ Architecture

```
Data Sources          Pipeline Stages              Storage & Serving
──────────────        ───────────────              ─────────────────
AWS Docs    ──┐
Spark Docs  ──┤→ ingest.py → preprocess.py → embed.py → store.py → Pinecone DB
K8s Docs    ──┤                                                          │
arXiv       ──┘                                                          │
PDF Files   ──┐                                               retrieve.py│
               └──────────────────────────────────────────────────┐      │
                                                              api/main.py │
                                                              (FastAPI)   │
                                                                   │      │
                                                              ui/app.py   │
                                                             (Streamlit)◄─┘
```

| Component | Technology | Purpose |
|---|---|---|
| `ingest.py` | requests, BeautifulSoup | Crawls web docs and PDFs |
| `preprocess.py` | Pure Python | Cleans text, creates overlapping chunks |
| `embed.py` | SentenceTransformers | Converts text → 384-dim vectors |
| `store.py` | Pinecone SDK | Upserts and queries vector index |
| `retrieve.py` | — | Combines embed + query in one call |
| `api/main.py` | FastAPI | REST API with `/query` and `/health` |
| `ui/app.py` | Streamlit | Chat interface |

---

## 📁 Project Structure

```
vector-pipeline/
├── src/
│   └── vector_pipeline/
│       ├── __init__.py
│       ├── config.py          # All settings loaded from .env
│       ├── ingest.py          # Web crawler + PDF + arXiv loaders
│       ├── preprocess.py      # Text cleaner + chunker
│       ├── embed.py           # SentenceTransformer wrapper
│       ├── store.py           # Pinecone upsert + query
│       ├── retrieve.py        # High-level retrieve() helper
│       └── run_ingest.py      # CLI entry point
├── api/
│   └── main.py                # FastAPI app
├── ui/
│   └── app.py                 # Streamlit chatbot
├── airflow/
│   └── dags/
│       └── refresh_dag.py     # Daily refresh DAG (3 AM)
├── lambda/
│   └── handler.py             # AWS Lambda serverless handler
├── spark/
│   └── embed_job.py           # PySpark large-scale embedding job
├── tests/
│   └── test_chunking.py       # Smoke tests
├── data/
│   └── raw/                   # Local document storage
├── config.example.env         # Environment variable template
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10, 3.11, or 3.12
- A free [Pinecone](https://pinecone.io) account

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/vector-pipeline.git
cd vector-pipeline

python -m venv .venv
source .venv/bin/activate          # Windows (Git Bash): source .venv/Scripts/activate

pip install requests beautifulsoup4 lxml pypdf python-dotenv \
    sentence-transformers pinecone fastapi "uvicorn[standard]" \
    pydantic streamlit tenacity tqdm
```

### 2. Configure environment

```bash
cp config.example.env .env
```

Edit `.env` with your values:

```env
PINECONE_API_KEY=your-key-here
PINECONE_INDEX=tech-docs
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384
```

### 3. Run the ingest pipeline

```bash
export PYTHONPATH=src

# Start small to verify
python -m vector_pipeline.run_ingest --source aws --limit 10

# Full ingest (all 3 sources)
python -m vector_pipeline.run_ingest --source aws --limit 100
python -m vector_pipeline.run_ingest --source spark --limit 100
python -m vector_pipeline.run_ingest --source k8s --limit 100
```

### 4. Start the API server (Terminal 1)

```bash
PYTHONPATH=src uvicorn api.main:app --reload --port 8000
```

Verify at: http://localhost:8000/health → `{"status":"ok"}`
Interactive API docs: http://localhost:8000/docs

### 5. Start the chatbot UI (Terminal 2)

```bash
PYTHONPATH=src streamlit run ui/app.py
```

Opens at: http://localhost:8501

---

## 📦 Supported Data Sources

| Source | Command | Description |
|---|---|---|
| AWS Docs | `--source aws` | Lambda, S3, EC2 documentation |
| Apache Spark | `--source spark` | Spark SQL, streaming, ML docs |
| Kubernetes | `--source k8s` | Pods, deployments, services docs |
| arXiv Papers | `--source arxiv --query "RAG" --limit 20` | Research paper abstracts |
| Local PDFs | `--source pdf --pdf ./my-docs/` | Your own PDF files |

---

## 🔌 API Reference

### `POST /query`

```json
{
  "query": "How do Lambda functions work?",
  "top_k": 5,
  "source": "aws"
}
```

Response:
```json
{
  "query": "How do Lambda functions work?",
  "took_ms": 142.5,
  "matches": [
    {
      "id": "abc123",
      "score": 0.921,
      "text": "AWS Lambda is a serverless compute service...",
      "source": "aws",
      "url": "https://docs.aws.amazon.com/lambda/...",
      "title": "AWS Lambda Developer Guide"
    }
  ]
}
```

### `GET /health`

```json
{"status": "ok"}
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `PINECONE_API_KEY` | — | Your Pinecone API key (required) |
| `PINECONE_INDEX` | `tech-docs` | Pinecone index name |
| `PINECONE_CLOUD` | `aws` | Cloud provider for serverless |
| `PINECONE_REGION` | `us-east-1` | Region for serverless index |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `EMBEDDING_DIM` | `384` | Vector dimensions |
| `CHUNK_TOKENS` | `400` | Max words per chunk |
| `CHUNK_OVERLAP` | `60` | Overlapping words between chunks |
| `UPSERT_BATCH_SIZE` | `100` | Vectors per Pinecone batch |

---

## 🧪 Running Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

---

## 📅 Automation

### Airflow (Daily Refresh at 3 AM)

```bash
export AIRFLOW_HOME=~/airflow
airflow db migrate
cp airflow/dags/refresh_dag.py ~/airflow/dags/
airflow scheduler &
airflow webserver --port 8080
```

### AWS Lambda (Serverless)

```bash
pip install -r requirements.txt --target ./package
cp -r src/vector_pipeline ./package/
cp lambda/handler.py ./package/
cd package && zip -r ../lambda_deploy.zip . && cd ..
```

Upload `lambda_deploy.zip` to AWS Lambda. Set Lambda memory to 1024 MB+. Trigger via EventBridge with `{"source": "aws", "limit": 20}`.

### PySpark (100k+ docs)

```bash
spark-submit --master local[*] spark/embed_job.py \
  --input data/raw/*.txt --output data/embeddings.parquet --upsert
```

---

## 🛠️ Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: vector_pipeline` | Run `export PYTHONPATH=src` first |
| `PINECONE_API_KEY is not set` | Make sure `.env` exists and you're in the project root |
| `pinecone-client` rename error | Run `pip uninstall pinecone-client -y && pip install pinecone` |
| First ingest is very slow | Normal — AI model downloads once (~90 MB), cached after |
| Port 8000 already in use | Change port: `uvicorn api.main:app --port 8001` |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Siddharth**
- GitHub: [SiddharthF18](https://github.com/SiddharthF18)

---

*Built with ❤️ using Python, FastAPI, Streamlit, SentenceTransformers, and Pinecone*
