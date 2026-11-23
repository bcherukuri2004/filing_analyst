"""Measure how well each search setup finds the right passage.

A question file has one JSON object per line:
  {"question": "...", "ticker": "AAPL", "year": 2023, "gold": ["383,285", "383.3"]}

A question counts as found when one of the top k passages is from the right
company and year and contains at least one of the gold strings (normally the
number the answer depends on). Plain string matching keeps the grading
objective: no model decides whether the search was right."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from .embed import Embedder
from .retrieve import Retriever
from .store import Hit, PgStore, Store


@dataclass
class Question:
    question: str
    ticker: str
    year: int
    gold: list[str]


def load_questions(path: str) -> list[Question]:
    out = []
    with open(path) as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = json.loads(line)
            if not d.get("gold"):
                raise ValueError(f"{path}:{n}: every question needs at least one gold string")
            out.append(Question(d["question"], d["ticker"].upper(), int(d["year"]), [str(g) for g in d["gold"]]))
    return out


def is_relevant(h: Hit, q: Question) -> bool:
    return h.ticker == q.ticker and h.fiscal_year == q.year and any(g in h.text for g in q.gold)


def score_ranking(hits: list[Hit], q: Question, k: int) -> tuple[bool, float]:
    """(found in top k, reciprocal rank within the top k)."""
    for rank, h in enumerate(hits[:k], start=1):
        if is_relevant(h, q):
            return True, 1.0 / rank
    return False, 0.0


def evaluate(questions: list[Question], search: Callable[[str], list[Hit]], k: int) -> dict:
    found, rr, n = 0, 0.0, len(questions)
    misses = []
    for q in questions:
        ok, r = score_ranking(search(q.question), q, k)
        found += ok
        rr += r
        if not ok:
            misses.append(q.question)
    return {"n": n, f"recall@{k}": found / n if n else 0.0, f"mrr@{k}": rr / n if n else 0.0, "misses": misses}


def run_eval(path: str, store: Store | PgStore, embedder: Embedder, k: int = 5, with_rerank: bool = True,
             log: Callable[[str], None] = print) -> list[dict]:
    questions = load_questions(path)
    # A question whose gold text is nowhere in its filing cannot be graded fairly
    # (the number may be written differently than expected), so it is left out and reported.
    gradable = [q for q in questions if store.any_contains(q.ticker, q.year, q.gold)]
    if len(gradable) < len(questions):
        log(f"Left out {len(questions) - len(gradable)} of {len(questions)} questions whose answer text is not in the indexed filing.")
    questions = gradable
    reranker = None
    if with_rerank:
        from .retrieve import CrossEncoderReranker

        reranker = CrossEncoderReranker()
    r = Retriever(store, embedder, reranker)
    ranking = "ts_rank" if store.kind == "postgres" else "BM25"  # how each store ranks keyword matches
    setups = [
        (f"keyword search only ({ranking})", dict(mode="bm25", rerank=False, use_filters=False)),
        ("vector search only", dict(mode="dense", rerank=False, use_filters=False)),
        ("hybrid (keyword + vector)", dict(mode="hybrid", rerank=False, use_filters=False)),
        ("hybrid + company/year filter", dict(mode="hybrid", rerank=False, use_filters=True)),
    ]
    if with_rerank:
        setups.append(("hybrid + filter + reranker", dict(mode="hybrid", rerank=True, use_filters=True)))
    rows = []
    log(f"{len(questions)} questions, top {k}\n")
    log(f"{'setup':<34} {'recall@' + str(k):>9} {'MRR@' + str(k):>9}")
    for name, opts in setups:
        res = evaluate(questions, lambda text, o=opts: r.search(text, k=k, **o)[0], k)
        res["setup"] = name
        rows.append(res)
        log(f"{name:<34} {res[f'recall@{k}']:>9.3f} {res[f'mrr@{k}']:>9.3f}")
    return rows
