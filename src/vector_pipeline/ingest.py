"""Data ingestion: web docs, arXiv, and PDFs.

Each loader yields normalized ``Document`` records with the raw text and
metadata (source, url, title, fetched_at). The downstream stages don't care
where the document came from.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

USER_AGENT = "vector-pipeline/1.0 (+https://example.com)"

# Curated seed URLs for the recommended tech-documentation dataset.
SOURCE_SEEDS: dict[str, list[str]] = {
    "aws": [
        "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
        "https://docs.aws.amazon.com/s3/latest/userguide/Welcome.html",
    ],
    "spark": [
        "https://spark.apache.org/docs/latest/",
        "https://spark.apache.org/docs/latest/sql-programming-guide.html",
    ],
    "k8s": [
        "https://kubernetes.io/docs/concepts/overview/",
        "https://kubernetes.io/docs/concepts/workloads/pods/",
    ],
}


@dataclass
class Document:
    id: str
    text: str
    source: str
    url: str = ""
    title: str = ""
    fetched_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


def _doc_id(source: str, key: str) -> str:
    return hashlib.sha1(f"{source}:{key}".encode()).hexdigest()[:16]


def fetch_html(url: str, timeout: int = 20) -> tuple[str, str]:
    """Return (title, visible_text) for an HTML page."""
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else url)
    text = " ".join(soup.get_text(" ").split())
    return title, text


def crawl_docs(
    source: str,
    seeds: Iterable[str] | None = None,
    max_pages: int = 25,
) -> Iterator[Document]:
    """Breadth-first crawl restricted to the seed's host."""
    seeds = list(seeds) if seeds else SOURCE_SEEDS.get(source, [])
    if not seeds:
        raise ValueError(f"No seeds configured for source '{source}'")

    seen: set[str] = set()
    queue: list[str] = list(seeds)
    host = urlparse(seeds[0]).netloc

    while queue and len(seen) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            title, text = fetch_html(url)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Failed to fetch %s: %s", url, exc)
            continue
        if len(text) < 200:
            continue
        yield Document(
            id=_doc_id(source, url),
            text=text,
            source=source,
            url=url,
            title=title,
        )
        # discover new links on the same host
        try:
            html = requests.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=15
            ).text
            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                nxt = urljoin(url, a["href"]).split("#")[0]
                if urlparse(nxt).netloc == host and nxt not in seen:
                    queue.append(nxt)
        except Exception:  # pragma: no cover
            continue


def load_pdf(path: str | Path, source: str = "pdf") -> Document:
    """Load a local PDF file as a single Document."""
    from pypdf import PdfReader

    p = Path(path)
    reader = PdfReader(str(p))
    text = " ".join((page.extract_text() or "") for page in reader.pages)
    text = " ".join(text.split())
    return Document(
        id=_doc_id(source, str(p)),
        text=text,
        source=source,
        url=str(p),
        title=p.stem,
    )


def fetch_arxiv(query: str, max_results: int = 10) -> Iterator[Document]:
    """Lightweight arXiv abstract loader using the public Atom API."""
    import xml.etree.ElementTree as ET

    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={requests.utils.quote(query)}&max_results={max_results}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (
            entry.findtext("a:summary", default="", namespaces=ns) or ""
        ).strip()
        link = entry.findtext("a:id", default="", namespaces=ns) or ""
        if not summary:
            continue
        yield Document(
            id=_doc_id("arxiv", link),
            text=f"{title}\n\n{summary}",
            source="arxiv",
            url=link,
            title=title,
        )
