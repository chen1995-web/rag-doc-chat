"""Answer generation: demo mode (free) and provider mode (paid).

Demo mode is deliberately *not* a pretend model. It returns the retrieved
evidence verbatim with an explicit "I found no answer in the documents" guard
when the ranking is weak. That is honest, it is free, and it makes the whole
pipeline testable end-to-end before you spend a cent on tokens.

The guardrail (confidence threshold + refusal) is the feature customers actually
care about: a system that confidently invents a number is worse than one that
says "I don't know". The threshold is tuned against scripts/eval.py, not guessed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from retrieve import Hit


@dataclass
class Answer:
    text: str
    confidence: float
    refused: bool
    mode: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float


CONFIDENCE_FLOOR = 0.15
# Calibrated against the score distribution in reports/eval-report.md:
# in-domain queries score vector>=0.05 and BM25>=2.1; out-of-domain queries
# score BM25=0 and vector<0.10. Vector and BM25 use different scales, so we
# normalise each to [0,1] and take the stronger signal -- OR-logic, because
# either ranker finding real evidence is enough to justify answering.
VEC_SCALE = 0.30
BM25_SCALE = 8.0


def _confidence(hits: list[Hit]) -> float:
    if not hits:
        return 0.0
    vec = max(0.0, hits[0].vector_score) / VEC_SCALE
    bm25 = max(0.0, hits[0].bm25_score) / BM25_SCALE
    return round(min(1.0, max(vec, bm25)), 4)


def build_context(hits: list[Hit], max_chars: int = 6000) -> str:
    blocks: list[str] = []
    used = 0
    for h in hits:
        block = f"[c{len(blocks)+1}] {h.title} (page {h.page_no})\n{h.content}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


SYSTEM_PROMPT = (
    "You answer strictly from the numbered context blocks.\n"
    "Rules:\n"
    "1. Cite with [cN] after every claim. If the context does not contain the "
    "answer, reply exactly: I couldn't find that in the provided documents.\n"
    "2. Never invent numbers, names, dates or figures.\n"
    "3. Answer in the same language the question uses.\n"
    "4. Be concise: lead with the answer, then the reason, then the citation."
)


def generate(question: str, hits: list[Hit], *, model: str, base_url: str,
             key: str, temperature: float = 0.0, max_context_chars: int = 6000) -> Answer:
    confidence = _confidence(hits)
    context = build_context(hits, max_context_chars)

    if not hits or confidence < CONFIDENCE_FLOOR:
        return Answer(
            text="I couldn't find that in the provided documents. "
                 "Try uploading more relevant files, or rephrase using terms that appear in them.",
            confidence=confidence, refused=True, mode="demo",
            model=model, prompt_tokens=0, completion_tokens=0, cost=0.0,
        )

    if model == "demo" or not key:
        return _demo_answer(question, hits, confidence, context)

    return _provider_answer(question, context, hits, confidence, model, base_url,
                            key, temperature)


def _demo_answer(question: str, hits: list[Hit], confidence: float,
                 context: str) -> Answer:
    """Evidence-grounded extractive answer -- free, deterministic, auditable."""
    best = hits[0]
    parts: list[str] = []
    parts.append(f"**Answer (from {best.title}, page {best.page_no}):**")
    sent = best.content.strip().replace("\n", " ")
    if len(sent) > 500:
        sent = sent[:500] + "..."
    parts.append(sent)
    if len(hits) > 1:
        others = ", ".join(f"{h.title} (p.{h.page_no})" for h in hits[1:4])
        parts.append(f"Supporting: {others}.")
    parts.append("")
    parts.append("> Demo mode returns retrieved evidence verbatim. Set `RAG_MODEL` "
                 "and `RAG_KEY` to get a generated answer.")
    text = "\n".join(parts)
    prompt_tokens = max(1, len(context) // 4)
    completion_tokens = max(1, len(text) // 4)
    return Answer(text=text, confidence=confidence, refused=False, mode="demo",
                  model="demo", prompt_tokens=prompt_tokens,
                  completion_tokens=completion_tokens, cost=0.0)


def _provider_answer(question: str, context: str, hits: list[Hit], confidence: float,
                     model: str, base_url: str, key: str, temperature: float) -> Answer:
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    text = resp.choices[0].message.content or ""
    pt = resp.usage.prompt_tokens if resp.usage else max(1, len(context) // 4)
    ct = resp.usage.completion_tokens if resp.usage else max(1, len(text) // 4)
    from usage import cost_for

    return Answer(text=text, confidence=confidence, refused=False, mode="provider",
                  model=model, prompt_tokens=pt, completion_tokens=ct,
                  cost=cost_for(model, pt, ct))
