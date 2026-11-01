"""Hybrid retrieval: dense vector similarity + BM25 keyword + RRF fusion.

This is the part that separates the demo from every "just wrap an embedding
API" tutorial. Two independent rankers disagree about what a query means, so
the answer is to fuse their *rankings*, not their scores -- Reciprocal Rank
Fusion is robust to each side having a different scale, and it is what makes
a name like "Q3 revenue" resolve to the page that literally says Q3 rather than
the page that shares the word "revenue" a lot.

Vector side: chunks are L2-normalised, so cosine = dot product.
Keyword side: BM25 over the same tokenizer the embedder uses.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from embeddings import tokenize


@dataclass
class Hit:
    doc_id: str
    title: str
    source: str
    idx: int
    page_no: int
    content: str
    chars: int
    vector_rank: int
    bm25_rank: int
    rrf: float
    vector_score: float
    bm25_score: float

    def as_citation(self) -> dict:
        return {
            "id": f"{self.doc_id}#{self.idx}",
            "doc_id": self.doc_id,
            "title": self.title,
            "page": self.page_no,
            "source": self.source,
            "snippet": self.content[:220],
            "rrf": round(self.rrf, 5),
        }


def _bm25_scores(query_tokens: list[str], doc_tokens: list[list[str]],
                 k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Score every document against the query tokens.

    Chinese is tokenised per character, which means raw frequency matching is
    dominated by stopwords (的/是/元/天). We drop any term that appears in more
    than ``max_df`` of the corpus before scoring -- that is a discriminative-
    signal filter, not a hardcoded stopword list, so it adapts as the corpus
    changes.
    """
    n = len(doc_tokens)
    if n == 0:
        return np.zeros(0)

    lengths = np.array([len(d) for d in doc_tokens], dtype=np.float64)
    avg_len = float(lengths.mean()) if lengths.mean() > 0 else 1.0
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))

    max_df = 0.6
    scores = np.zeros(n, dtype=np.float64)
    for term in set(query_tokens):
        if df[term] == 0 or df[term] / n > max_df:
            continue
        # Per-document term frequency. Getting this wrong collapses BM25 to a
        # length-only ranking, which is exactly the failure this eval caught.
        f = np.array([t.count(term) for t in doc_tokens], dtype=np.float64)
        idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        denom = f + k1 * (1 - b + b * lengths / avg_len)
        scores += idf * (f * (k1 + 1)) / np.where(denom == 0, 1e-9, denom)
    return scores


def _rrf(rank_a: dict[int, int], rank_b: dict[int, int], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for rank_map in (rank_a, rank_b):
        for idx, rank in rank_map.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    def __init__(self, rows: list[dict], dim: int) -> None:
        """rows: list of chunk dicts with keys id, doc_id, title, source, idx, page_no, content, chars, embed(bytes|None)."""
        self.rows = rows
        self.dim = dim
        vecs = []
        for r in rows:
            arr = np.frombuffer(r["embed"], dtype=np.float32) if r["embed"] else None
            vecs.append(arr)
        self.matrix = np.stack([v if v is not None else np.zeros(dim, dtype=np.float32) for v in vecs]) if vecs else np.zeros((0, dim), dtype=np.float32)
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix / np.where(norms == 0, 1.0, norms)
        self.token_docs = [tokenize(r["content"]) for r in rows]
        self.id_of = {id(r): i for i, r in enumerate(rows)}

    @property
    def size(self) -> int:
        return len(self.rows)

    def search(self, query: str, doc_ids: list[str] | None = None,
               top_k: int = 6, vector_pool: int = 25, bm25_pool: int = 25,
               rrf_k: int = 60) -> list[Hit]:
        if self.size == 0:
            return []

        mask = np.ones(self.size, dtype=bool)
        if doc_ids:
            allowed = set(doc_ids)
            mask = np.array([r["doc_id"] in allowed for r in self.rows], dtype=bool)
        if not mask.any():
            return []

        q_vec = self._embed_query(query)
        sims = self.matrix @ q_vec
        sims = np.where(mask, sims, -np.inf)

        q_tokens = tokenize(query)
        bm25 = _bm25_scores(q_tokens, self.token_docs) if self.size else np.zeros(0)
        bm25 = np.where(mask, bm25, -np.inf)

        # rank fusion only over the restricted pool
        pool_n = max(vector_pool, bm25_pool)
        vec_idx = np.argsort(-sims, kind="stable")[:pool_n]
        bm_idx = np.argsort(-bm25, kind="stable")[:pool_n]

        vec_rank = {int(i): rank + 1 for rank, i in enumerate(vec_idx) if np.isfinite(sims[i])}
        bm_rank = {int(i): rank + 1 for rank, i in enumerate(bm_idx) if np.isfinite(bm25[i])}
        fused = _rrf(vec_rank, bm_rank, rrf_k)

        ranked = sorted(fused, key=lambda i: -fused[i])[:top_k]
        hits: list[Hit] = []
        for i in ranked:
            r = self.rows[i]
            hits.append(Hit(
                doc_id=r["doc_id"], title=r["title"], source=r["source"],
                idx=r["idx"], page_no=r["page_no"], content=r["content"],
                chars=r["chars"],
                vector_rank=vec_rank.get(i, 0), bm25_rank=bm_rank.get(i, 0),
                rrf=fused[i],
                vector_score=float(sims[i]) if np.isfinite(sims[i]) else 0.0,
                bm25_score=float(bm25[i]) if np.isfinite(bm25[i]) else 0.0,
            ))
        return hits

    def _embed_query(self, query: str) -> np.ndarray:
        # Late import so the retriever stays usable without a configured provider.
        from embeddings import get_embedder
        from config import load

        embedder = get_embedder(load())
        vec = embedder.embed([query])[0]
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec
