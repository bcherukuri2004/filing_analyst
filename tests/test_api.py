"""Tests for the web API (filing_analyst/api.py).

They use the small FICTIONAL filings from fixtures.py, the instant
HashingEmbedder and a fake language model, so they need no network, no
model download and no API key. Besides the normal cases they try to break
the API with bad input, a failing language model and parallel requests."""
import re
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from filing_analyst.api import create_app

ACME_REVENUE_Q = "What was Acme's total net revenue in 2023?"


class FakeLLM:
    """Stands in for Claude. Answers with the revenue figure and cites the
    passage that really contains it, like a well behaved model would."""

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if self.error:
            raise self.error
        if self.reply is not None:
            return self.reply
        for number, text in re.findall(r"\[(\d+)\] (.*?)(?=\n\n\[\d+\] |\n\nQuestion:)", user, re.S):
            if "1,250,400" in text:
                return f"Total net revenue was $1,250,400 thousand [{number}]."
        return "I could not find this in the filings I have."


@pytest.fixture
def client(retriever):
    return TestClient(create_app(retriever=retriever, llm=FakeLLM()))


@pytest.fixture
def client_without_llm(retriever):
    return TestClient(create_app(retriever=retriever, llm=None))


# ------------------------------------------------------------ health

def test_health_reports_what_is_loaded(client, store):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["passages"] == store.count() > 0
    assert body["ask_enabled"] is True
    assert body["reranker"] is False  # the test retriever has none


def test_health_says_when_ask_is_off(client_without_llm):
    assert client_without_llm.get("/health").json()["ask_enabled"] is False


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404


# ------------------------------------------------------------ search

def test_search_finds_the_revenue_table(client):
    r = client.post("/search", json={"question": ACME_REVENUE_Q})
    assert r.status_code == 200
    body = r.json()
    assert body["filters"] == {"tickers": ["ACME"], "year": 2023}
    results = body["results"]
    assert 1 <= len(results) <= 5
    assert [x["rank"] for x in results] == list(range(1, len(results) + 1))
    assert all(x["ticker"] == "ACME" and x["fiscal_year"] == 2023 for x in results)
    assert any("1,250,400" in x["text"] for x in results)
    first = results[0]
    for key in ("label", "company", "item", "section", "kind", "score"):
        assert key in first


def test_search_respects_k(client):
    r = client.post("/search", json={"question": ACME_REVENUE_Q, "k": 2})
    assert r.status_code == 200
    assert len(r.json()["results"]) <= 2


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_search_every_mode_works(client, mode):
    r = client.post("/search", json={"question": ACME_REVENUE_Q, "mode": mode})
    assert r.status_code == 200
    assert r.json()["results"]


def test_search_without_filters_reports_none(client):
    r = client.post("/search", json={"question": ACME_REVENUE_Q, "use_filters": False})
    assert r.status_code == 200
    assert r.json()["filters"] == {"tickers": [], "year": None}


def test_search_question_with_search_syntax_is_safe(client):
    # Quotes, colons, stars and parentheses mean something to the keyword engine.
    r = client.post("/search", json={"question": 'revenue" OR * NEAR( "x": -y', "mode": "bm25"})
    assert r.status_code == 200


def test_search_question_with_no_matches_returns_empty_list(client):
    r = client.post("/search", json={"question": "zzzqqq xxyyzz", "mode": "bm25"})
    assert r.status_code == 200
    assert r.json()["results"] == []


@pytest.mark.parametrize("payload", [
    {},                                                  # no question at all
    {"question": ""},                                    # empty
    {"question": "    \n\t "},                           # only whitespace
    {"question": "x" * 1001},                            # too long
    {"question": 12345},                                 # not text
    {"question": ACME_REVENUE_Q, "k": 0},                # too few results
    {"question": ACME_REVENUE_Q, "k": 21},               # too many results
    {"question": ACME_REVENUE_Q, "k": -3},
    {"question": ACME_REVENUE_Q, "k": "five"},
    {"question": ACME_REVENUE_Q, "mode": "magic"},       # unknown search mode
])
def test_search_rejects_bad_input(client, payload):
    r = client.post("/search", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_search_rejects_a_body_that_is_not_json(client):
    r = client.post("/search", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_search_handles_parallel_requests(client):
    # A real server answers many people at once. Each request must still work.
    def one(_):
        return client.post("/search", json={"question": ACME_REVENUE_Q}).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(one, range(24)))
    assert codes == [200] * 24


# ------------------------------------------------------------ ask

def test_ask_returns_a_checked_answer_with_citations(client):
    r = client.post("/ask", json={"question": ACME_REVENUE_Q})
    assert r.status_code == 200
    body = r.json()
    assert "1,250,400" in body["answer"]
    assert body["verified"] is True and body["problems"] == []
    assert body["refused"] is False
    assert len(body["citations"]) == 1
    assert body["citations"][0]["label"].startswith("Acme Robotics")


def test_ask_flags_an_invented_number(retriever):
    llm = FakeLLM(reply="Total net revenue was $9,999,999 thousand [1].")
    r = TestClient(create_app(retriever=retriever, llm=llm)).post("/ask", json={"question": ACME_REVENUE_Q})
    assert r.status_code == 200
    body = r.json()
    assert body["verified"] is False
    assert any("9999999" in p for p in body["problems"])


def test_ask_passes_a_refusal_through(retriever):
    llm = FakeLLM(reply="I could not find this in the filings I have.")
    r = TestClient(create_app(retriever=retriever, llm=llm)).post("/ask", json={"question": ACME_REVENUE_Q})
    assert r.status_code == 200
    assert r.json()["refused"] is True
    assert r.json()["citations"] == []


def test_ask_without_a_model_is_503(client_without_llm):
    r = client_without_llm.post("/ask", json={"question": ACME_REVENUE_Q})
    assert r.status_code == 503
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_ask_when_the_model_fails_is_502_and_hides_the_error(retriever):
    llm = FakeLLM(error=RuntimeError("secret internal detail sk-ant-123"))
    r = TestClient(create_app(retriever=retriever, llm=llm)).post("/ask", json={"question": ACME_REVENUE_Q})
    assert r.status_code == 502
    assert "sk-ant" not in r.text and "secret" not in r.text


def test_ask_rejects_bad_input_before_calling_the_model(retriever):
    llm = FakeLLM()
    c = TestClient(create_app(retriever=retriever, llm=llm))
    assert c.post("/ask", json={"question": "   "}).status_code == 422
    assert c.post("/ask", json={"question": ACME_REVENUE_Q, "k": 50}).status_code == 422
    assert llm.calls == 0  # no money spent on bad requests
