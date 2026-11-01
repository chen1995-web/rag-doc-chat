"""Text extraction for pdf / txt / md / html.

``pypdf`` is the only optional import here: the rest of the pipeline is
dependency-light, and ``scripts/seed.py`` deliberately ships plain-text
documents so the demo never depends on a PDF library being present.
"""

from __future__ import annotations

import os
import re

PAGE_RE = re.compile(r"^\s*-{3,}\s*page\s+\d+\s*-{3,}$", re.I | re.M)


def _extract_pdf(path: str) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install pypdf: pip install pypdf") from exc

    reader = PdfReader(path)
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def _extract_html(path: str) -> list[tuple[int, str]]:
    from bs4 import BeautifulSoup

    with open(path, encoding="utf-8", errors="ignore") as fh:
        soup = BeautifulSoup(fh.read(), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return [(1, soup.get_text("\n"))]


def _strip_page_markers(text: str) -> str:
    return PAGE_RE.sub("", text).strip()


def extract(path: str) -> list[tuple[int, str]]:
    """Return ``[(page_no, text), ...]`` for any supported file."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext in {".txt", ".md", ".markdown", ".rst"}:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            return [(1, _strip_page_markers(fh.read()))]
    if ext in {".html", ".htm"}:
        return _extract_html(path)
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return [(1, _strip_page_markers(fh.read()))]
