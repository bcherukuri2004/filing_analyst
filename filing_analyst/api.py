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
    FILING_ANALYST_INDEX           index folder (default "index")
    FILING_ANALYST_RERANK          set to 0 to skip loading the reranker
    ANTHROPIC_API_KEY              turns on /ask
    FILING_ANALYST_API_KEY         if set, /search and /ask need it in an X-API-Key header
    FILING_ANALYST_ASK_PER_MINUTE  /ask calls allowed per client per minute (default 10)
    FILING_ANALYST_ASK_GLOBAL_PER_MINUTE
                                   /ask calls allowed per minute for everyone together (default 60)
"""
from __future__ import annotations

import hmac
import ipaddress
import logging
import math
import os
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
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
    auth_required: bool


# ------------------------------------------------------------ settings and protection

MIN_KEY_CHARS = 16
RATE_WINDOW_SECONDS = 60


def settings_from_env(env: Mapping[str, str]) -> dict:
    """Read the protection settings. A bad value stops the server at startup
    with a clear message, instead of running with protection quietly off."""
    key = (env.get("FILING_ANALYST_API_KEY") or "").strip() or None  # empty counts as unset
    if key is not None and len(key) < MIN_KEY_CHARS:
        raise ValueError(f"FILING_ANALYST_API_KEY must be at least {MIN_KEY_CHARS} characters long")
    return {"api_key": key,
            "ask_per_minute": _limit_from_env(env, "FILING_ANALYST_ASK_PER_MINUTE", 10),
            "ask_global_per_minute": _limit_from_env(env, "FILING_ANALYST_ASK_GLOBAL_PER_MINUTE", 60)}


def _limit_from_env(env: Mapping[str, str], name: str, default: int) -> int:
    """A limit must be a whole number of at least 1. There is deliberately no
    value that means "no limit", so a typo cannot switch protection off."""
    raw = env.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def client_key(host: str) -> str:
    """Who to count a request against, for the per-client limit.

    IPv4 addresses count one by one. IPv6 addresses are grouped by their first
    64 bits (the "/64" network), because one home or phone connection is
    usually given a whole /64, which is 2^64 addresses: counting each one
    separately would let one person look like billions of clients.
    Anything that is not an IP address (like "unknown") is used as it is."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:  # "::ffff:10.0.0.1" is really the IPv4 address 10.0.0.1
            return str(ip.ipv4_mapped)
        # Built from the number, not the text, so a zone like "fe80::1%eth0" cannot upset it.
        return str(ipaddress.IPv6Network((int(ip) >> 64 << 64, 64)))
    return str(ip)


MAX_TRACKED_CLIENTS = 10_000


class RateLimiter:
    """At most `limit` calls per client in any 60 second window. It remembers
    the time of each client's recent calls and forgets them after a minute.
    Kept in memory, so it resets when the server restarts and is per process.

    Memory is bounded: at most `max_clients` clients are remembered, each with
    at most `limit` times. The table is kept in order of when each client was
    last seen; when it is full, the client seen longest ago is dropped. Every
    call does a fixed amount of work however many clients there are.

    Trade-off: if a flood of new addresses fills the table, a quiet client can
    be dropped and its count forgotten. A client that keeps calling (even if it
    is being refused) stays at the "recent" end and keeps its count."""

    def __init__(self, limit: int, clock: Callable[[], float], max_clients: int = MAX_TRACKED_CLIENTS,
                 message: str | None = None):
        if max_clients < 1:
            raise ValueError(f"max_clients must be at least 1, got {max_clients}")
        self.limit = limit
        self.clock = clock
        self.max_clients = max_clients
        self.message = message or f"Too many /ask requests. Limit is {limit} per minute."
        self.calls: OrderedDict[str, deque[float]] = OrderedDict()
        self.lock = threading.Lock()

    def undo(self, client: str) -> None:
        """Take back the last call counted for this client (used when a later
        check refuses the request, so the client is not charged for it)."""
        with self.lock:
            recent = self.calls.get(client)
            if recent:
                recent.pop()

    def check(self, client: str) -> None:
        """Count one call, or raise 429 if the client is over the limit."""
        with self.lock:
            now = self.clock()
            recent = self.calls.get(client)
            if recent is None:
                if len(self.calls) >= self.max_clients:
                    self.calls.popitem(last=False)  # drop the client seen longest ago
                recent = self.calls[client] = deque()
            else:
                self.calls.move_to_end(client)  # mark as seen just now, refused or not
            while recent and recent[0] <= now - RATE_WINDOW_SECONDS:
                recent.popleft()
            if len(recent) >= self.limit:
                wait = math.ceil(recent[0] + RATE_WINDOW_SECONDS - now)
                raise HTTPException(429, self.message, headers={"Retry-After": str(max(1, wait))})
            recent.append(now)


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


def create_app(retriever: Retriever | None = None, llm: LLM | None = None, *, api_key: str | None = None,
               ask_per_minute: int | None = None, ask_global_per_minute: int | None = None,
               clock: Callable[[], float] = time.monotonic) -> FastAPI:
    """Build the app. Tests pass in a small retriever and a fake model; the
    real server passes nothing and everything is loaded at startup.

    api_key: when set, /search and /ask require it in the X-API-Key header.
    ask_per_minute: when set, each client may call /ask this many times a minute.
    ask_global_per_minute: when set, /ask may be called this many times a minute
        by everyone together. This is the ceiling on language model spending.
    clock: where the rate limit gets the time (tests use a fake one)."""
    state = _State()
    state.retriever, state.llm = retriever, llm
    limiter = RateLimiter(ask_per_minute, clock) if ask_per_minute else None
    # One shared counter for the whole server, so it only ever tracks one "client".
    global_limiter = RateLimiter(
        ask_global_per_minute, clock, max_clients=1,
        message=f"The server is busy: /ask is limited to {ask_global_per_minute} calls a minute "
                "for everyone together. Try again shortly.") if ask_global_per_minute else None
    expected = api_key.encode() if api_key else None

    def require_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        if expected is None:
            return
        # compare_digest takes the same time however much of the key is right,
        # so the key cannot be guessed one character at a time by timing replies.
        if x_api_key is None or not hmac.compare_digest(x_api_key.encode("utf-8", "replace"), expected):
            raise HTTPException(401, "Missing or wrong API key. Send it in the X-API-Key header.",
                                headers={"WWW-Authenticate": "X-API-Key"})
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
        return Health(status="ok", passages=passages, embedder=embedder, reranker=r.reranker is not None,
                      ask_enabled=state.llm is not None, auth_required=expected is not None)

    @app.post("/search", response_model=SearchResponse, dependencies=[Depends(require_key)])
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

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_key)])
    def ask(req: AskRequest, request: Request) -> AskResponse:
        # By the time this runs, the key and the request body have passed their
        # checks, so rejected requests never count against the limit.
        if state.llm is None:
            raise HTTPException(503, "Answers are turned off. Set ANTHROPIC_API_KEY and restart the server. "
                                     "/search works without it.")
        # The caller's own limit is checked first, so a caller who is over it
        # never uses up the shared allowance. If the shared allowance is then
        # full, the caller's call is taken back: being turned away because the
        # server is busy does not count against them.
        who = client_key(request.client.host if request.client else "unknown")
        if limiter:
            limiter.check(who)
        if global_limiter:
            try:
                global_limiter.check("everyone")
            except HTTPException:
                if limiter:
                    limiter.undo(who)
                raise
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
# Bad protection settings fail right here, before the server accepts anything.
app = create_app(**settings_from_env(os.environ))
