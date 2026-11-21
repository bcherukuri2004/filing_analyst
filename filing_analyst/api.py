"""Web API for filing_analyst: the same search and answers as the command
line, over HTTP, so other programs (a web page, a chat bot, a teammate's
script) can use them.

    pip install -e ".[api]"
    uvicorn filing_analyst.api:app --port 8000
    # then open http://localhost:8000/docs for an interactive page

Endpoints:
    GET  /health   is the server up, and what is loaded
    POST /search   best matching passages for a question (free, no API key)
    POST /ask      a written answer with checked citations (needs ANTHROPIC_API_KEY)

Settings, read from environment variables when the server starts:
    FILING_ANALYST_INDEX   index folder (default "index")
    FILING_ANALYST_RERANK  set to 0 to skip loading the reranker
    ANTHROPIC_API_KEY      turns on /ask
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .answer import LLM, answer_question
from .retrieve import Retriever

log = logging.getLogger("filing_analyst.api")

MAX_QUESTION_CHARS = 1000
MAX_K = 20


# ------------------------------------------------------------ request and response shapes
# FastAPI checks every request against these before our code runs, and
# answers 422 with an explanation when something is wrong.

class _QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    k: int = Field(5, ge=1, le=MAX_K, description="how many passages to use")

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question is empty")
        return v


class SearchRequest(_QuestionRequest):
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    rerank: bool = True
    use_filters: bool = Field(True, description="read company and year from the question")


class AskRequest(_QuestionRequest):
    pass


class Passage(BaseModel):
    rank: int
    label: str
    ticker: str
    company: str
    fiscal_year: int
    item: str
    section: str
    kind: str
    text: str
    score: float


class FiltersOut(BaseModel):
    tickers: list[str]
    year: int | None


class SearchResponse(BaseModel):
    question: str
    filters: FiltersOut
    reranked: bool
    results: list[Passage]


class Citation(BaseModel):
    number: int
    label: str


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    verified: bool
    problems: list[str]
    citations: list[Citation]


class Health(BaseModel):
    status: str
    passages: int
    embedder: str | None
    reranker: bool
    ask_enabled: bool


# ------------------------------------------------------------ the app

class _State:
    retriever: Retriever | None = None
    llm: LLM | None = None


def _load_from_environment(state: _State) -> None:
    """Load the index and models once, when the server starts. Loading takes
    seconds; a search takes milliseconds, so it must not happen per request."""
    from .embed import FastEmbedder
    from .store import Store

    folder = os.environ.get("FILING_ANALYST_INDEX", "index")
    store = Store(folder)
    if store.count() == 0:
        raise RuntimeError(f"The index in '{folder}' is empty. Run: python -m filing_analyst ingest <folder>")
    reranker = None
    if os.environ.get("FILING_ANALYST_RERANK", "1") != "0":
        from .retrieve import CrossEncoderReranker

        reranker = CrossEncoderReranker()
    state.retriever = Retriever(store, FastEmbedder(), reranker)
    if os.environ.get("ANTHROPIC_API_KEY"):
        from .answer import AnthropicLLM

        state.llm = AnthropicLLM()
    log.info("loaded %d passages from %s; reranker=%s; ask=%s",
             store.count(), folder, reranker is not None, state.llm is not None)


def create_app(retriever: Retriever | None = None, llm: LLM | None = None) -> FastAPI:
    """Build the app. Tests pass in a small retriever and a fake model; the
    real server passes nothing and everything is loaded at startup."""
    state = _State()
    state.retriever, state.llm = retriever, llm
    # One database connection and one copy of the models are shared by all
    # requests, which the server runs on several threads at once. The lock lets
    # one search use them at a time. The slow model call in /ask happens
    # outside the lock so it does not hold up other people's searches.
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state.retriever is None:
            _load_from_environment(state)
        yield

    app = FastAPI(
        title="filing_analyst",
        description="Search SEC 10-K annual reports and get answers with checked citations.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def loaded_retriever() -> Retriever:
        if state.retriever is None:  # only possible if startup was skipped
            raise HTTPException(503, "The search index is not loaded.")
        return state.retriever

    @app.get("/health", response_model=Health)
    def health() -> Health:
        r = loaded_retriever()
        with lock:
            passages, embedder = r.store.count(), r.store.embedder_name()
        return Health(status="ok", passages=passages, embedder=embedder,
                      reranker=r.reranker is not None, ask_enabled=state.llm is not None)

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        r = loaded_retriever()
        with lock:
            hits, filters = r.search(req.question, k=req.k, mode=req.mode, rerank=req.rerank,
                                     use_filters=req.use_filters)
        results = [
            Passage(rank=i, label=h.label, ticker=h.ticker, company=h.company, fiscal_year=h.fiscal_year,
                    item=h.item, section=h.section, kind=h.kind, text=h.text, score=h.score)
            for i, h in enumerate(hits, start=1)
        ]
        return SearchResponse(question=req.question, filters=FiltersOut(tickers=filters.tickers, year=filters.year),
                              reranked=req.rerank and r.reranker is not None, results=results)

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        if state.llm is None:
            raise HTTPException(503, "Answers are turned off. Set ANTHROPIC_API_KEY and restart the server. "
                                     "/search works without it.")
        r = loaded_retriever()
        with lock:
            hits, _ = r.search(req.question, k=req.k)
        try:
            ans = answer_question(req.question, hits, state.llm)
        except Exception:
            # The full error goes to the server log only. It can hold details
            # (keys, internal addresses) that callers must never see.
            log.exception("language model call failed")
            raise HTTPException(502, "The language model could not be reached. Try again later.") from None
        citations = [] if ans.refused else [Citation(number=n, label=hits[n - 1].label) for n in ans.cited]
        return AskResponse(question=req.question, answer=ans.text, refused=ans.refused,
                           verified=ans.refused or ans.verified, problems=ans.problems, citations=citations)

    return app


# What `uvicorn filing_analyst.api:app` runs. Nothing heavy happens here: the
# index and models load when the server starts, not when this file is imported.
app = create_app()
