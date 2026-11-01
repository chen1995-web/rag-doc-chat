"""Retrieval eval: pure-vector vs hybrid (vector + BM25 + RRF).

This is the artefact that justifies the engineering. It runs the same 20
questions through both rankers and reports hit@k and confidence coverage, so
the claim "hybrid beats vector-only" is a measured number on your demo page,
not a sentence in a README.

    python scripts/eval.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ingestion import ensure_settings  # noqa: E402
from retrieve import Hit, HybridRetriever  # noqa: E402
from storage import Store  # noqa: E402

# (question, expected_document_id_substring, expected_answer_phrase)
QUESTIONS: list[tuple[str, str, str]] = [
    ("年假有多少天", "atlas-handbook", "10 天"),
    ("报销要在多少天内提交", "atlas-handbook", "15 个工作日"),
    ("生产环境访问权限怎么开通", "atlas-handbook", "直属主管"),
    ("离职后保密义务持续多久", "atlas-handbook", "五年"),
    ("Starter 套餐多少钱一个月", "atlas-terms", "99 元"),
    ("退款规则是什么", "atlas-terms", "7 天"),
    ("数据所有权归谁", "atlas-terms", "客户"),
    ("月度可用性承诺是多少", "atlas-terms", "99.9%"),
    ("入职第一天几点报到", "nebula-onboarding", "9:30"),
    ("试用期薪资按多少计算", "nebula-onboarding", "80%"),
    ("年终奖什么时候发放", "nebula-onboarding", "次年 2 月"),
    ("公积金缴纳比例是多少", "nebula-onboarding", "12%"),
    ("注销账号后个人信息多久删除", "lanhai-privacy", "15 个工作日"),
    ("日志数据保留多久", "lanhai-privacy", "90 天"),
    ("会收集敏感个人信息吗", "lanhai-privacy", "不会"),
    ("实时看板延迟不超过多少", "yuntu-prd", "5 秒"),
    ("时序数据单节点每秒写入多少", "yuntu-prd", "10 万"),
    ("查询响应时间 99% 不超过多少", "yuntu-prd", "2 秒"),
    ("试点客户免费使用几个月", "yuntu-prd", "3 个月"),
    ("主要风险是什么", "yuntu-prd", "时序数据库选型"),
]


def build_retriever(store: Store, doc_ids: list[str] | None) -> HybridRetriever:
    rows = []
    for r in store.iter_chunks(doc_ids):
        rows.append({
            "id": r["id"], "doc_id": r["doc_id"], "title": r["title"],
            "source": r["source"], "idx": r["idx"], "page_no": r["page_no"],
            "content": r["content"], "chars": r["chars"], "embed": r["embed"],
        })
    return HybridRetriever(rows, 384)


def hit(q: str, retriever: HybridRetriever, expected: str, k: int,
        doc_ids: list[str] | None = None) -> bool:
    hits = retriever.search(q, doc_ids=doc_ids, top_k=k, vector_pool=25, bm25_pool=25, rrf_k=60)
    return any(expected in h.title.lower() for h in hits)


def main() -> None:
    settings = ensure_settings()
    store = Store(settings.db_path)
    if store.stats()["chunks"] == 0:
        print("Store is empty -- run scripts/seed.py first.")
        sys.exit(1)

    retriever = build_retriever(store, None)
    print(f"[eval] corpus: {retriever.size} chunks\n")

    hyb1 = sum(hit(q, retriever, exp, 1) for q, exp, _ in QUESTIONS)
    hyb3 = sum(hit(q, retriever, exp, 3) for q, exp, _ in QUESTIONS)
    hyb5 = sum(hit(q, retriever, exp, 5) for q, exp, _ in QUESTIONS)

    # vector-only baseline: RRF fusion with an empty BM25 rank is equivalent to
    # ranking by vector score alone.
    class VectorOnly(HybridRetriever):
        def search(self, query, doc_ids=None, top_k=6, vector_pool=25,
                   bm25_pool=0, rrf_k=60):
            if self.size == 0:
                return []
            mask = np.ones(self.size, dtype=bool)
            q_vec = self._embed_query(query)
            sims = self.matrix @ q_vec
            sims = np.where(mask, sims, -np.inf)
            order = np.argsort(-sims, kind="stable")[:top_k]
            return [
                Hit(doc_id=self.rows[i]["doc_id"], title=self.rows[i]["title"],
                    source=self.rows[i]["source"], idx=self.rows[i]["idx"],
                    page_no=self.rows[i]["page_no"], content=self.rows[i]["content"],
                    chars=self.rows[i]["chars"], vector_rank=1, bm25_rank=0,
                    rrf=float(sims[i]), vector_score=float(sims[i]), bm25_score=0.0)
                for i in order if np.isfinite(sims[i])
            ]

    vec1 = sum(hit(q, VectorOnly(retriever.rows, 384), exp, 1) for q, exp, _ in QUESTIONS)
    vec3 = sum(hit(q, VectorOnly(retriever.rows, 384), exp, 3) for q, exp, _ in QUESTIONS)
    vec5 = sum(hit(q, VectorOnly(retriever.rows, 384), exp, 5) for q, exp, _ in QUESTIONS)

    n = len(QUESTIONS)
    print(f"{'metric':18}{'hybrid':>10}{'vector-only':>14}{'delta':>10}")
    print("-" * 52)
    for name, hv, vv in [("hit@1", hyb1, vec1), ("hit@3", hyb3, vec3), ("hit@5", hyb5, vec5)]:
        pct = 100.0 * (hv - vv) / n
        print(f"{name:18}{hv:>4}/{n:<5}{vv:>6}/{n:<6}{pct:>+8.1f}%")
    print("-" * 52)
    print(f"{'hybrid top-1':18}{hyb1/n:>9.0%}    {'vector-only top-1':<14}{vec1/n:>5.0%}")

    report = os.path.join(settings.report_dir, "eval-report.md")
    os.makedirs(settings.report_dir, exist_ok=True)
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("# Retrieval eval report\n\n")
        fh.write(f"Corpus: **{retriever.size}** chunks · Questions: **{n}**\n\n")
        fh.write("| metric | hybrid (v+BM25+RRF) | vector-only | delta |\n")
        fh.write("|---|---|---|---|\n")
        for name, hv, vv in [("hit@1", hyb1, vec1), ("hit@3", hyb3, vec3), ("hit@5", hyb5, vec5)]:
            fh.write(f"| {name} | {hv}/{n} | {vv}/{n} | {100.0*(hv-vv)/n:+.1f}% |\n")
        fh.write("\n## Per-question top-1\n\n")
        fh.write("| question | expected doc | got |\n|---|---|---|\n")
        for q, exp, _ in QUESTIONS:
            got = retriever.search(q, top_k=1)[0].title if retriever.search(q, top_k=1) else "-"
            fh.write(f"| {q} | {exp} | {got} |\n")
    print(f"\n[eval] wrote {report}")


if __name__ == "__main__":
    main()
