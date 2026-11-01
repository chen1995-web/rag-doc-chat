# RAG Doc Chat — grounded document Q&A with measured retrieval

**Live demo:** `python app.py` → http://127.0.0.1:8000 · **Login:** none (open) · **Source:** full, no paywall

Upload PDFs / TXT / MD / HTML, ask questions in natural language, get answers
that cite the exact document and page they came from — and refuse to answer
when the documents genuinely don't contain it.

## The problem it solves

Every "chat with your PDF" demo answers with confidence and no accountability.
A support team that pastes a customer question into a 400-page contract needs
to know *which page* the answer came from, and a CFO who just approved the
project needs to know *what each question costs*. This project answers both:
citations by page, and a per-request cost meter.

## What I built

- **Hybrid retrieval** — dense vector similarity + BM25 keyword + Reciprocal
  Rank Fusion. Measured against a 20-question eval set: **hit@1 20/20 (100%)**
  vs **18/20 (90%)** for vector-only — a **+10%** lift, reported by
  `scripts/eval.py`, not asserted.
- **Grounded answers with page citations** — every claim traces to `[cN]`, the
  UI renders citation chips that open the source snippet.
- **Guardrail that refuses instead of hallucinating** — confidence is calibrated
  on the eval distribution (in-domain vs out-of-domain); a query about the
  weather returns "I couldn't find that", not a fabricated forecast.
- **Per-request cost accounting** — token counts × a price table surfaced in
  the UI and summed in the sidebar, so "what does this cost" is one click.
- **Bilingual UI (i18n)** — English by default, one-click 中文 toggle in the
  header; choice persists in `localStorage` and `<html lang>` follows it.
- **Zero-cost demo mode** — local feature-hash embeddings + extractive answers
  mean the whole pipeline runs with no API key and no spend. Set `RAG_MODEL` +
  `RAG_KEY` to switch to a real model; the eval harness measures the delta.

## Trade-offs I made

- **Local hash embeddings, not a neural model.** They rank by lexical overlap
  and are deterministic and free. They are *not* semantic — that is exactly why
  the eval exists: it tells you the cost of the trade-off in hit-rate terms.
- **RRF over score fusion.** Fusing ranks is robust to the two rankers having
  wildly different scales; fusing raw scores is not.
- **Citations at chunk/page level, not sentence level.** Page-level is honest
  about what PDF extraction can actually promise (text order is not guaranteed).
- **SQLite, not Postgres+pgvector.** Zero infra, single file, and the similarity
  scan is a numpy matmul over a few thousand vectors. Swap to pgvector when the
  corpus stops fitting in memory — the retriever interface doesn't change.

## Run it locally

```bash
pip install -r requirements.txt
python scripts/seed.py          # loads data/uploads into data/rag.sqlite3
python app.py                   # http://127.0.0.1:8000
```

Use a real model (optional):

```bash
export RAG_MODEL=gpt-4o-mini
export RAG_KEY=sk-...
export RAG_EMBED_PROVIDER=openai
```

## Metrics

Measured with `scripts/eval.py` on 5 docs / 13 chunks, 20 questions:

| metric | hybrid (vector+BM25+RRF) | vector-only | delta |
|---|---|---|---|
| hit@1 | 20/20 | 18/20 | **+10.0%** |
| hit@3 | 20/20 | 19/20 | +5.0% |
| hit@5 | 20/20 | 20/20 | 0.0% |

Report written to `reports/eval-report.md`.

## Layout

```
config.py        env-driven settings + model price table
embeddings.py    hash vs OpenAI embedders; CJK bigram tokenisation
chunking.py      overlap chunking with page provenance
pdf_extract.py   pdf/txt/md/html extraction
storage.py       SQLite: documents, chunks, embed cache, usage
retrieve.py      hybrid retriever + RRF fusion + BM25
llm.py           demo vs provider answers + calibrated guardrail
ingestion.py     file → pages → chunks → embeddings, with dedup cache
app.py           FastAPI: upload, chat, stats
static/           chat UI (EN default + 中文 toggle, localStorage)
scripts/seed.py  corpus loader
scripts/eval.py  retrieval eval (the artefact that sells it)
```
