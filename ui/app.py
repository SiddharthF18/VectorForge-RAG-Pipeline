"""Streamlit chatbot UI that talks to the FastAPI retrieval endpoint.

Run with: streamlit run ui/app.py
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("RETRIEVAL_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Tech Docs RAG", page_icon=":books:", layout="wide")
st.title("Tech Docs RAG Chatbot")
st.caption(f"Backed by Pinecone + SentenceTransformers · API: {API_URL}")

with st.sidebar:
    top_k = st.slider("Top K", 1, 15, 5)
    source = st.selectbox("Filter source", ["(all)", "aws", "spark", "k8s", "arxiv", "pdf"])
    st.markdown("---")
    st.markdown("Set `RETRIEVAL_API_URL` to point at a remote API.")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        for m in turn.get("matches", []):
            with st.expander(f"{m['title'] or m['source']} (score {m['score']:.3f})"):
                st.write(m["text"])
                if m["url"]:
                    st.markdown(f"[Source]({m['url']})")

prompt = st.chat_input("Ask anything about the indexed docs...")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {"query": prompt, "top_k": top_k}
    if source != "(all)":
        payload["source"] = source

    try:
        resp = requests.post(f"{API_URL}/query", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        matches = data.get("matches", [])
        answer_lines = [
            f"**Top {len(matches)} passages** ({data.get('took_ms', 0):.0f} ms)"
        ]
        for i, m in enumerate(matches, 1):
            snippet = m["text"][:300] + ("..." if len(m["text"]) > 300 else "")
            answer_lines.append(f"\n**{i}. {m['title'] or m['source']}** — {snippet}")
        answer = "\n".join(answer_lines) if matches else "No matches found."
    except Exception as exc:
        answer = f"Request failed: {exc}"
        matches = []

    with st.chat_message("assistant"):
        st.markdown(answer)
        for m in matches:
            with st.expander(f"{m['title'] or m['source']} (score {m['score']:.3f})"):
                st.write(m["text"])
                if m["url"]:
                    st.markdown(f"[Source]({m['url']})")

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "matches": matches}
    )
