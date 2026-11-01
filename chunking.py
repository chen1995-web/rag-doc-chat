"""Chunking with page-level provenance.

Citations are the whole point of a document Q&A system: without ``page_no``
you can only say "here is a plausible answer", and a plausible answer is what
users screenshot and send to your boss. So every chunk remembers which page it
came from, and the UI links the citation chip back to that page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARA_SPLIT = re.compile(r"\n{2,}")
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass
class Chunk:
    doc_id: str
    idx: int
    page_no: int
    content: str
    chars: int


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in _PARA_SPLIT.split(text)]
    return [p for p in parts if p]


def pack(units: list[str], size: int, overlap: int) -> list[str]:
    """Greedy packing of paragraph units into size-bounded chunks.

    ``overlap`` characters are carried over from the previous chunk so a
    sentence that straddles a boundary still appears in at least one complete
    chunk.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        blob = "\n\n".join(current)
        if blob.strip():
            chunks.append(blob.strip())
        tail = blob[-overlap:] if overlap and blob else ""
        current = [tail] if tail else []
        current_len = len(tail)

    for unit in units:
        unit_len = len(unit)
        if current_len + unit_len + 2 > size and current:
            flush()
        current.append(unit)
        current_len += unit_len + 2
    if current:
        flush()
    return [c for c in chunks if c]


def chunk_pages(doc_id: str, pages: list[tuple[int, str]], size: int, overlap: int) -> list[Chunk]:
    """Turn extracted ``(page_no, text)`` pairs into provenance-tagged chunks."""
    out: list[Chunk] = []
    idx = 0
    for page_no, raw in pages:
        units = _split_paragraphs(raw)
        if not units:
            continue
        # A page whose single paragraph is far over ``size`` gets sentence-split.
        units = [u if len(u) <= size else s for u in units for s in _SENT_SPLIT.split(u) if s.strip()]
        for text in pack(units, size, overlap):
            out.append(Chunk(doc_id, idx, page_no, text, len(text)))
            idx += 1
    return out
