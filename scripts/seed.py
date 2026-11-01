"""Seed the corpus: ingest data/uploads into the SQLite store.

    python scripts/seed.py

Idempotent: re-running replaces the existing documents rather than duplicating
them, so it is safe to call from a fresh checkout.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ingestion import ensure_settings, ingest_directory  # noqa: E402
from storage import Store  # noqa: E402


def main() -> None:
    settings = ensure_settings()
    store = Store(settings.db_path)
    results = ingest_directory(store, settings, settings.upload_dir)
    print(f"[seed] ingested {len(results)} documents into {settings.db_path}")
    for r in results:
        print(f"  {r.title:24} pages={r.pages:2} chunks={r.chunks:3} chars={r.chars:6} cached={r.cached}")
    print(f"[seed] store stats: {store.stats()}")


if __name__ == "__main__":
    main()
