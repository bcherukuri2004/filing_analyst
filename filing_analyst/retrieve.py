"""Finding the right passages: keyword search, vector search, the two merged
with reciprocal rank fusion, optional reranking, and company and year
filters read from the question."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .embed import Embedder
from .store import Hit, PgStore, Store

RRF_K = 60  # standard constant for reciprocal rank fusion
POOL = 50  # how many results each method contributes before merging
RERANK_POOL = 30  # how many merged results the reranker looks at

_SUFFIXES = re.compile(
    r"[,\.]?\s+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings|group|llc|lp|n\.v|s\.a)\.?$", re.I
)
_GENERIC_FIRST_WORDS = {"the", "united", "american", "general", "first", "national", "international", "global"}


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """A small cross encoder that reads the question and a passage together
    and scores how well the passage answers it. Slower than vector search but
    much better at ordering a short list."""

    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, texts)]


@dataclass
class Filters:
    tickers: list[str]
    year: int | None

    def __bool__(self) -> bool:
        return bool(self.tickers or self.year)


def company_aliases(ticker: str, company: str) -> list[str]:
    name = _SUFFIXES.sub("", company.strip())
    aliases = {name.lower()}
    first = name.split()[0].lower() if name.split() else ""
    if len(first) >= 4 and first not in _GENERIC_FIRST_WORDS:
        aliases.add(first)
    return sorted(aliases, key=len, reverse=True)


def detect_filters(question: str, companies: list[tuple[str, str]], years_for) -> Filters:
    """Which companies and fiscal year does the question mention? Without this,
    "What was Apple's revenue in 2023?" can be answered with a passage from
    some other company's filing that happens to use the same words."""
    tickers: list[str] = []
    for ticker, company in companies:
        hit = re.search(rf"\b{re.escape(ticker)}\b", question)  # tickers are matched case sensitively
        if not hit:
            for alias in company_aliases(ticker, company):
                if re.search(rf"\b{re.escape(alias)}(?:'s|’s)?\b", question, re.I):
                    hit = True
                    break
        if hit and ticker not in tickers:
            tickers.append(ticker)
    year = None
    m = re.findall(r"\b(20\d\d)\b", question)
    if m:
        cand = int(m[-1])
        known = set()
        for t in tickers or [t for t, _ in companies]:
            known.update(years_for(t))
        if cand in known:
            year = cand
    return Filters(tickers, year)


def rrf(lists: list[list[Hit]], k: int = RRF_K) -> list[Hit]:
    """Reciprocal rank fusion: a passage's score is the sum of 1/(k+rank)
    over every list it appears in. Needs no tuning of score scales, which
    differ wildly between keyword and vector search."""
    scores: dict[int, float] = {}
    keep: dict[int, Hit] = {}
    for hits in lists:
        for rank, h in enumerate(hits, start=1):
            scores[h.id] = scores.get(h.id, 0.0) + 1.0 / (k + rank)
            keep.setdefault(h.id, h)
    ordered = sorted(scores, key=lambda i: -scores[i])
    return [Hit(**{**keep[i].__dict__, "score": scores[i]}) for i in ordered]


class Retriever:
    def __init__(self, store: Store | PgStore, embedder: Embedder | None = None, reranker: Reranker | None = None):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker

    def _search_one(self, question: str, mode: str, ticker: str | None, year: int | None) -> list[Hit]:
        lists = []
        if mode in ("bm25", "hybrid"):
            lists.append(self.store.keyword_search(question, POOL, ticker, year))
        if mode in ("dense", "hybrid"):
            if self.embedder is None:
                raise RuntimeError("dense search needs an embedder")
            lists.append(self.store.dense_search(self.embedder.embed_query(question), POOL, ticker, year))
        return lists[0] if len(lists) == 1 else rrf(lists)

    def search(self, question: str, k: int = 5, mode: str = "hybrid", rerank: bool = True,
               use_filters: bool = True) -> tuple[list[Hit], Filters]:
        filters = Filters([], None)
        if use_filters:
            filters = detect_filters(question, self.store.companies(), self.store.years)
        tickers = filters.tickers or [None]
        per_ticker = [self._search_one(question, mode, t, filters.year) for t in tickers]
        # For a question about several companies, interleave so each gets a fair share.
        merged = per_ticker[0] if len(per_ticker) == 1 else rrf(per_ticker)
        if rerank and self.reranker and merged:
            pool = merged[:RERANK_POOL]
            scores = self.reranker.score(question, [f"{h.label}\n{h.text}" for h in pool])
            order = sorted(range(len(pool)), key=lambda i: -scores[i])
            merged = [Hit(**{**pool[i].__dict__, "score": scores[i]}) for i in order]
        return merged[:k], filters
