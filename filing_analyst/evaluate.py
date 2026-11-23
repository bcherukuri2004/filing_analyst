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
    group: str = ""  # the kind of question, e.g. "revenue", for the breakdown


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
            out.append(Question(d["question"], d["ticker"].upper(), int(d["year"]), [str(g) for g in d["gold"]],
                                str(d.get("group", ""))))
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
    by_group: dict[str, list[int]] = {}  # group: [found, asked]
    for q in questions:
        ok, r = score_ranking(search(q.question), q, k)
        found += ok
        rr += r
        g = by_group.setdefault(q.group, [0, 0])
        g[0] += ok
        g[1] += 1
        if not ok:
            misses.append(q.question)
    return {"n": n, f"recall@{k}": found / n if n else 0.0, f"mrr@{k}": rr / n if n else 0.0, "misses": misses,
            "by_group": {g: (f, a) for g, (f, a) in by_group.items()}}


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
    groups = sorted({q.group for q in questions if q.group})
    if groups:
        # One average can hide a whole kind of question failing, so show each kind.
        short = ["keyword", "vector", "hybrid", "+filter", "+rerank"][:len(rows)]
        log(f"\nrecall@{k} by kind of question\n")
        log(f"{'kind':<26} {'n':>3} " + " ".join(f"{s:>8}" for s in short))
        for g in groups:
            asked = rows[0]["by_group"][g][1]
            log(f"{g:<26} {asked:>3} " + " ".join(f"{r['by_group'][g][0] / asked:>8.2f}" for r in rows))
    return rows


# ------------------------------------------------------------ the gate
# A gate file says how good search must be, for example:
#
#   {"min_questions": 35,
#    "setups": {"vector search only": {"recall@5": 0.95, "mrr@5": 0.75}}}
#
# `eval --gate gate.json` exits with an error if any line is missed, which is
# what makes CI fail when a change makes search worse.

def load_gate(path: str) -> dict:
    """Read a gate file, refusing one that is malformed or checks nothing."""
    try:
        with open(path) as f:
            gate = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from None
    if not isinstance(gate, dict):
        raise ValueError(f"{path} must hold a JSON object")
    n = gate.get("min_questions")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"{path}: min_questions must be a whole number of at least 1")
    setups = gate.get("setups")
    if not isinstance(setups, dict) or not setups or not all(isinstance(m, dict) and m for m in setups.values()):
        raise ValueError(f"{path}: a gate that checks nothing would always pass; list setups and their minimums")
    for setup, minimums in setups.items():
        for metric, value in minimums.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{path}: {setup} {metric} must be a number between 0 and 1, got {value!r}")
    return gate


def check_gate(rows: list[dict], gate: dict) -> list[str]:
    """Every way the results miss the gate, or an empty list if they pass."""
    problems = []
    by_setup = {r["setup"]: r for r in rows}
    graded = min((r["n"] for r in rows), default=0)
    if graded < gate["min_questions"]:
        problems.append(f"only {graded} question{'s' if graded != 1 else ''} could be graded, "
                        f"the gate needs {gate['min_questions']} (did parsing break?)")
    for setup, minimums in gate["setups"].items():
        row = by_setup.get(setup)
        if row is None:
            problems.append(f"the gate checks '{setup}', which this eval did not run")
            continue
        for metric, minimum in minimums.items():
            if metric not in row:
                problems.append(f"the gate checks {metric} for '{setup}', which this eval did not report")
            elif row[metric] < minimum:
                problems.append(f"{setup}: {metric} is {row[metric]:.3f}, the gate needs at least {minimum}")
    return problems
