"""Smoke tests for the deterministic stages (no network, no model load)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vector_pipeline.ingest import Document  # noqa: E402
from vector_pipeline.preprocess import chunk_documents, chunk_text, clean_text  # noqa: E402


def test_clean_text_collapses_whitespace_and_strips_boilerplate():
    raw = "Hello   world\n\n  cookie policy: we use cookies"
    assert clean_text(raw) == "Hello world"


def test_chunk_text_respects_size_and_overlap():
    words = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_text(words, chunk_tokens=100, overlap=20)
    assert len(chunks) > 1
    # overlap: last 20 words of chunk N should equal first 20 of chunk N+1
    a = chunks[0].split()[-20:]
    b = chunks[1].split()[:20]
    assert a == b


def test_chunk_documents_yields_chunk_objects():
    doc = Document(id="d1", text=" ".join("hello" for _ in range(800)),
                   source="test", url="x", title="t")
    chunks = list(chunk_documents([doc]))
    assert chunks
    assert all(c.doc_id == "d1" for c in chunks)
    assert all(c.text for c in chunks)


if __name__ == "__main__":
    test_clean_text_collapses_whitespace_and_strips_boilerplate()
    test_chunk_text_respects_size_and_overlap()
    test_chunk_documents_yields_chunk_objects()
    print("OK")
