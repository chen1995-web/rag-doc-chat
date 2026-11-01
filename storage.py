"""SQLite storage layer for documents, chunks, embeddings and usage.

Embeddings are stored as raw float32 blobs rather than JSON so a 200-document
corpus stays a few megabytes and similarity scans stay in numpy. The embed
cache deduplicates repeated chunks -- corporate PDFs are full of near-identical
boilerplate, and paying for embeddings on it is the classic way a free demo
turns into a surprise invoice.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'text',
    pages       INTEGER NOT NULL DEFAULT 0,
    chunks      INTEGER NOT NULL DEFAULT 0,
    chars       INTEGER NOT NULL DEFAULT 0,
    source      TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    page_no     INTEGER NOT NULL,
    content     TEXT NOT NULL,
    chars       INTEGER NOT NULL,
    embed       BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, idx);
CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL DEFAULT '',
    doc_ids     TEXT NOT NULL DEFAULT '',
    model       TEXT NOT NULL DEFAULT '',
    prompt_toks INTEGER NOT NULL DEFAULT 0,
    comp_toks   INTEGER NOT NULL DEFAULT 0,
    cost        REAL NOT NULL DEFAULT 0,
    confidence  REAL NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS embed_cache (
    key     TEXT PRIMARY KEY,
    dim     INTEGER NOT NULL,
    embed   BLOB NOT NULL
);
"""


def blob_to_vec(blob: bytes | None, dim: int) -> np.ndarray | None:
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size != dim:
        return None
    return arr


def vec_to_blob(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(vec, dtype=np.float32).tobytes()


def content_key(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


@dataclass
class DocMeta:
    id: str
    title: str
    kind: str
    pages: int
    chunks: int
    chars: int
    source: str
    created_at: float
    updated_at: float


class Store:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with self.conn() as db:
            db.executescript(SCHEMA)

    def conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    # -- documents -------------------------------------------------------
    def add_document(self, doc_id: str, title: str, kind: str, source: str,
                     pages: int, chunks: int, chars: int) -> None:
        now = time.time()
        with self.conn() as db:
            db.execute(
                "INSERT INTO documents(id,title,kind,source,pages,chunks,chars,created_at,updated_at)"
                " VALUES(?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET title=excluded.title, kind=excluded.kind,"
                " source=excluded.source, pages=excluded.pages, chunks=excluded.chunks,"
                " chars=excluded.chars, updated_at=excluded.updated_at",
                (doc_id, title, kind, source, pages, chunks, chars, now, now),
            )

    def delete_document(self, doc_id: str) -> None:
        with self.conn() as db:
            db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            db.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    def list_documents(self) -> list[DocMeta]:
        with self.conn() as db:
            rows = db.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [DocMeta(*r) for r in rows]

    def get_document(self, doc_id: str) -> DocMeta | None:
        with self.conn() as db:
            row = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return DocMeta(*row) if row else None

    # -- chunks ----------------------------------------------------------
    def add_chunks(self, doc_id: str, items: list[tuple[int, int, str, int, bytes | None]]) -> None:
        """items: (idx, page_no, content, chars, embed_blob)"""
        with self.conn() as db:
            db.executemany(
                "INSERT INTO chunks(doc_id,idx,page_no,content,chars,embed) VALUES(?,?,?,?,?,?)",
                [(doc_id, i, p, c, n, e) for (i, p, c, n, e) in items],
            )

    def iter_chunks(self, doc_ids: list[str] | None = None) -> list[sqlite3.Row]:
        sql = (
            "SELECT c.id, c.doc_id, c.idx, c.page_no, c.content, c.chars, c.embed,"
            " d.title AS title, d.source AS source"
            " FROM chunks c JOIN documents d ON d.id = c.doc_id"
        )
        args: tuple = ()
        if doc_ids:
            marks = ",".join("?" * len(doc_ids))
            sql += f" WHERE c.doc_id IN ({marks})"
            args = tuple(doc_ids)
        sql += " ORDER BY c.doc_id, c.idx"
        with self.conn() as db:
            return db.execute(sql, args).fetchall()

    # -- embed cache -----------------------------------------------------
    def cached_embed(self, key: str, dim: int) -> np.ndarray | None:
        with self.conn() as db:
            row = db.execute("SELECT embed FROM embed_cache WHERE key=?", (key,)).fetchone()
        return blob_to_vec(row["embed"], dim) if row else None

    def set_cached_embed(self, key: str, dim: int, vec: np.ndarray) -> None:
        with self.conn() as db:
            db.execute(
                "INSERT INTO embed_cache(key,dim,embed) VALUES(?,?,?)"
                " ON CONFLICT(key) DO UPDATE SET dim=excluded.dim, embed=excluded.embed",
                (key, dim, vec_to_blob(vec)),
            )

    # -- messages / usage ------------------------------------------------
    def log_message(self, *, mid: str, question: str, answer: str, doc_ids: list[str],
                    model: str, prompt_toks: int, comp_toks: int, cost: float,
                    confidence: float) -> None:
        with self.conn() as db:
            db.execute(
                "INSERT INTO messages(id,question,answer,doc_ids,model,prompt_toks,"
                "comp_toks,cost,confidence,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (mid, question, answer, json.dumps(doc_ids), model, prompt_toks,
                 comp_toks, cost, confidence, time.time()),
            )

    def stats(self) -> dict:
        with self.conn() as db:
            docs = db.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"]
            chunks = db.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
            chars = db.execute("SELECT COALESCE(SUM(chars),0) n FROM chunks").fetchone()["n"]
            msgs = db.execute("SELECT COUNT(*) n FROM messages").fetchone()["n"]
            cost = db.execute("SELECT COALESCE(SUM(cost),0) n FROM messages").fetchone()["n"]
        return {"documents": docs, "chunks": chunks, "chars": chars,
                "messages": msgs, "total_cost_usd": round(cost, 6)}
