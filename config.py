"""Runtime configuration for rag-doc-chat.

Everything is overridable through environment variables so the *same* codebase
runs in two modes:

* ``RAG_MODEL=demo`` (default) -- zero API key, local embeddings, extractive
  answers. The demo works on a laptop with no account and no money.
* ``RAG_MODEL=gpt-4o-mini`` + ``RAG_KEY`` -- the production path against any
  OpenAI-compatible ``/chat/completions`` endpoint.

The price table is what makes the "cost per answer" panel honest: the meter
multiplies token counts by *your* price, not a guess.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


# USD per 1,000 tokens (input / output).
MODEL_PRICES: dict[str, dict[str, float]] = {
    "demo": {"input": 0.0, "output": 0.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "claude-sonnet": {"input": 3.00, "output": 15.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "deepseek-chat": {"input": 0.27, "output": 1.10},
}


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000

    data_dir: str = "data"
    db_path: str = ""
    upload_dir: str = ""
    report_dir: str = ""

    # ingestion
    chunk_size: int = 800
    chunk_overlap: int = 120
    embed_batch_size: int = 16

    # hybrid retrieval
    retrieve_top_k: int = 6
    vector_pool: int = 25
    bm25_pool: int = 25
    rrf_k: int = 60

    # guardrail thresholds (tuned on the eval set in scripts/eval.py)
    min_confidence: float = 0.18
    min_rrf_score: float = 0.012

    # model
    model: str = "demo"
    base_url: str = "https://api.openai.com/v1"
    key: str = ""
    temperature: float = 0.0
    max_context_chars: int = 6000

    embed_provider: str = "local"
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 384

    @property
    def demo_mode(self) -> bool:
        return (not self.key) or self.model.lower() == "demo"


def load() -> Settings:
    s = Settings()
    s.host = os.getenv("RAG_HOST", s.host)
    s.port = _env_int("RAG_PORT", s.port)

    s.data_dir = os.getenv("RAG_DATA_DIR", s.data_dir)
    s.db_path = os.getenv("RAG_DB", os.path.join(s.data_dir, "rag.sqlite3"))
    s.upload_dir = os.getenv("RAG_UPLOAD_DIR", os.path.join(s.data_dir, "uploads"))
    s.report_dir = os.getenv("RAG_REPORT_DIR", "reports")

    s.chunk_size = _env_int("RAG_CHUNK_SIZE", s.chunk_size)
    s.chunk_overlap = _env_int("RAG_CHUNK_OVERLAP", s.chunk_overlap)
    s.embed_batch_size = _env_int("RAG_EMBED_BATCH", s.embed_batch_size)

    s.retrieve_top_k = _env_int("RAG_TOP_K", s.retrieve_top_k)
    s.vector_pool = _env_int("RAG_VECTOR_POOL", s.vector_pool)
    s.bm25_pool = _env_int("RAG_BM25_POOL", s.bm25_pool)
    s.rrf_k = _env_int("RAG_RRF_K", s.rrf_k)

    s.min_confidence = _env_float("RAG_MIN_CONFIDENCE", s.min_confidence)
    s.min_rrf_score = _env_float("RAG_MIN_RRF", s.min_rrf_score)

    s.model = os.getenv("RAG_MODEL", s.model)
    s.base_url = os.getenv("RAG_BASE_URL", s.base_url)
    s.key = os.getenv("RAG_KEY", "").strip()
    s.temperature = _env_float("RAG_TEMPERATURE", s.temperature)
    s.max_context_chars = _env_int("RAG_MAX_CONTEXT_CHARS", s.max_context_chars)

    s.embed_provider = os.getenv("RAG_EMBED_PROVIDER", s.embed_provider)
    s.embed_model = os.getenv("RAG_EMBED_MODEL", s.embed_model)
    s.embed_dim = _env_int("RAG_EMBED_DIM", s.embed_dim)
    return s
