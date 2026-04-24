"""FastAPI retrieval service.

Run with: uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from vector_pipeline.retrieve import retrieve

logger = logging.getLogger("vector_pipeline.api")

app = FastAPI(title="Vector Retrieval API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=25)
    source: Optional[str] = None


class Match(BaseModel):
    id: str
    score: float
    text: str
    source: str
    url: str
    title: str


class QueryResponse(BaseModel):
    query: str
    took_ms: float
    matches: list[Match]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def post_query(req: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    try:
        raw = retrieve(req.query, top_k=req.top_k, source=req.source)
    except Exception as exc:  # pragma: no cover
        logger.exception("retrieval failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    took = (time.perf_counter() - started) * 1000
    return QueryResponse(query=req.query, took_ms=round(took, 2), matches=raw)
