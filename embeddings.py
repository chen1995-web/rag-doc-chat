"""Embedding providers.

Two implementations matter here:

* ``LocalHashEmbedding`` -- deterministic feature hashing into a fixed-width
  vector. Zero API key, zero network, fully reproducible. This is what lets the
  demo boot with ``RAG_MODEL=demo`` on a laptop with no account.
* ``OpenAIEmbedding`` -- the production path: ``text-embedding-3-small`` or any
  OpenAI-compatible ``/embeddings`` endpoint.

The switch is measured, not vibes: ``scripts/eval.py`` reports hit@k for both
providers so you can see the quality delta before paying for anything.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Protocol

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Word tokenisation for latin, sliding-window bigrams for CJK.

    Single-character CJK tokenisation makes every ranker a stopword matcher:
    的/是/元/天 appear everywhere and drown the discriminative signal. Character
    bigrams (公积金, 金比, 比例) carry real meaning without needing a segmentation
    dictionary, and they make BM25 and the hash embedder agree on what a term is.
    """
    tokens: list[str] = []
    for run in TOKEN_RE.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", run):
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        else:
            tokens.append(run)
    return tokens


class EmbeddingProvider(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class LocalHashEmbedding:
    """Feature-hashing embedder.

    Each token is fingerprinted with md5 into ``dim`` buckets; a sign bit kills
    the collision bias of naive hashing. Vectors are L2-normalised so cosine
    similarity is a plain dot product.

    It is intentionally *not* a language model. It ranks by lexical overlap,
    which is enough to demo the whole pipeline end-to-end for free -- and it
    makes the hybrid-search comparison honest, because both passes see the
    same corpus.
    """

    name = "local-hash-384"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token, n in Counter(tokenize(text)).items():
                digest = hashlib.md5(token.encode("utf-8")).digest()
                for shift in range(0, 8, 2):  # four fingerprint bits per token
                    bucket = int.from_bytes(digest[shift : shift + 2], "big") % self.dim
                    sign = 1.0 if (digest[shift] & 0x80) else -1.0
                    out[row, bucket] += sign * n
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


class OpenAIEmbedding:
    name = "openai-text-embedding-3-small"

    def __init__(self, model: str, base_url: str, key: str, dim: int = 1536) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.key}"},
            json={"model": self.model, "input": texts},
            timeout=60.0,
        )
        resp.raise_for_status()
        vecs = np.asarray([d["embedding"] for d in resp.json()["data"]], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1.0, norms)


def get_embedder(settings) -> EmbeddingProvider:
    if settings.embed_provider == "openai" and settings.key:
        return OpenAIEmbedding(
            model=settings.embed_model,
            base_url=settings.base_url,
            key=settings.key,
            dim=settings.embed_dim,
        )
    return LocalHashEmbedding(dim=settings.embed_dim)
