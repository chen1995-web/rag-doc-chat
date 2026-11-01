"""FastAPI app: document upload, hybrid retrieval, grounded answers.

Run:  python app.py          (or uvicorn app:app --reload)
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import load
from ingestion import ensure_settings, ingest_file
from llm import generate
from retrieve import HybridRetriever
from storage import Store

app = FastAPI(title="RAG Doc Chat", version="1.0.0")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def get_store() -> Store:
    settings = ensure_settings()
    return Store(settings.db_path)


def get_retriever(store: Store, doc_ids: Optional[list[str]] = None) -> HybridRetriever:
    rows = []
    for r in store.iter_chunks(doc_ids):
        rows.append({
            "id": r["id"], "doc_id": r["doc_id"], "title": r["title"],
            "source": r["source"], "idx": r["idx"], "page_no": r["page_no"],
            "content": r["content"], "chars": r["chars"], "embed": r["embed"],
        })
    dim = 384
    return HybridRetriever(rows, dim)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/stats")
def stats() -> dict:
    settings = ensure_settings()
    store = Store(settings.db_path)
    data = store.stats()
    data["model"] = settings.model
    data["demo_mode"] = settings.demo_mode
    return data


@app.get("/api/documents")
def documents() -> dict:
    settings = ensure_settings()
    store = Store(settings.db_path)
    return {"documents": [vars(d) for d in store.list_documents()]}


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...),
                          title: Optional[str] = Form(None)) -> dict:
    settings = ensure_settings()
    store = Store(settings.db_path)
    suffix = os.path.splitext(file.filename or "upload.txt")[1]
    dest = os.path.join(settings.upload_dir, f"{uuid.uuid4().hex[:8]}{suffix}")
    with open(dest, "wb") as fh:
        fh.write(await file.read())
    try:
        result = ingest_file(store, settings, dest, title=title)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "doc_id": result.doc_id, "title": result.title, "pages": result.pages,
        "chunks": result.chunks, "chars": result.chars, "cached": result.cached,
    }


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    settings = ensure_settings()
    store = Store(settings.db_path)
    if not store.get_document(doc_id):
        raise HTTPException(status_code=404, detail="document not found")
    store.delete_document(doc_id)
    return {"deleted": doc_id}


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    doc_ids: Optional[list[str]] = None
    top_k: int = 6


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    settings = ensure_settings()
    store = Store(settings.db_path)
    retriever = get_retriever(store, req.doc_ids)
    hits = retriever.search(
        req.question, doc_ids=req.doc_ids, top_k=req.top_k,
        vector_pool=settings.vector_pool, bm25_pool=settings.bm25_pool,
        rrf_k=settings.rrf_k,
    )
    answer = generate(
        req.question, hits,
        model=settings.model, base_url=settings.base_url, key=settings.key,
        temperature=settings.temperature, max_context_chars=settings.max_context_chars,
    )
    store.log_message(
        mid=f"msg_{uuid.uuid4().hex[:8]}", question=req.question, answer=answer.text,
        doc_ids=[h.doc_id for h in hits], model=answer.model,
        prompt_toks=answer.prompt_tokens, comp_toks=answer.completion_tokens,
        cost=answer.cost, confidence=answer.confidence,
    )
    return {
        "answer": answer.text,
        "confidence": answer.confidence,
        "refused": answer.refused,
        "mode": answer.mode,
        "model": answer.model,
        "cost": round(answer.cost, 6),
        "tokens": {"prompt": answer.prompt_tokens, "completion": answer.completion_tokens},
        "citations": [h.as_citation() for h in hits],
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    settings = load()
    uvicorn.run(app, host=settings.host, port=settings.port)
