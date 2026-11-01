"""Ingestion pipeline: file -> pages -> chunks -> embeddings -> store.

The embed cache is the money-saver here. Corporate documents repeat
boilerplate -- every page ends with the same confidentiality footer, every
contract repeats "the Parties agree". Re-embedding identical text is the
quiet way a free demo becomes a real invoice.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from chunking import chunk_pages
from config import Settings, load
from embeddings import get_embedder
from pdf_extract import extract
from storage import Store, content_key, vec_to_blob

SUPPORTED = {".pdf", ".txt", ".md", ".markdown", ".rst", ".html", ".htm"}


@dataclass
class IngestResult:
    doc_id: str
    title: str
    pages: int
    chunks: int
    chars: int
    cached: int


def _new_doc_id() -> str:
    return f"doc_{uuid.uuid4().hex[:10]}"


def ingest_file(store: Store, settings: Settings, path: str,
                title: str | None = None, doc_id: str | None = None) -> IngestResult:
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "text"
    base = title or os.path.splitext(os.path.basename(path))[0]
    doc_id = doc_id or _new_doc_id()

    pages = extract(path)
    if not pages:
        raise ValueError(f"No extractable text in {path}")

    chunks = chunk_pages(doc_id, pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError(f"Chunking produced nothing for {path}")

    embedder = get_embedder(settings)
    dim = embedder.dim

    items: list[tuple[int, int, str, int, bytes | None]] = []
    cached = 0
    to_embed: list[tuple[int, str]] = []

    for ch in chunks:
        vec = store.cached_embed(content_key(ch.content), dim)
        if vec is not None:
            cached += 1
            items.append((ch.idx, ch.page_no, ch.content, ch.chars, vec_to_blob(vec)))
        else:
            to_embed.append((ch.idx, ch.content))

    if to_embed:
        vectors = embedder.embed([c for _, c in to_embed])
        for (idx, content), vec in zip(to_embed, vectors):
            store.set_cached_embed(content_key(content), dim, vec)
            page_no = next(ch.page_no for ch in chunks if ch.idx == idx)
            items.append((idx, page_no, content, len(content), vec_to_blob(vec)))

    chars = sum(ch.chars for ch in chunks)
    store.add_document(doc_id, base, ext, os.path.basename(path),
                       len(pages), len(chunks), chars)
    store.add_chunks(doc_id, items)
    return IngestResult(doc_id, base, len(pages), len(chunks), chars, cached)


def ingest_directory(store: Store, settings: Settings, directory: str,
                     title_prefix: str = "") -> list[IngestResult]:
    """Bulk ingest every supported file in a directory."""
    results: list[IngestResult] = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path) or os.path.splitext(name)[1].lower() not in SUPPORTED:
            continue
        results.append(ingest_file(store, settings, path,
                                   title=f"{title_prefix}{os.path.splitext(name)[0]}"))
    return results


def ensure_settings() -> Settings:
    s = load()
    os.makedirs(s.data_dir, exist_ok=True)
    os.makedirs(s.upload_dir, exist_ok=True)
    return s
