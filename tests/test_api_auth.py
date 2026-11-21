"""Tests for API key protection and the /ask rate limit.

/ask spends money on every call (it calls a paid language model), so a
server reachable by other machines must not let strangers use it. These
tests try the ways a caller might get in without the key, and check that
no model call (no cost) happens for any rejected request."""
import pytest
from fastapi.testclient import TestClient
from test_api import ACME_REVENUE_Q, FakeLLM

from filing_analyst.api import create_app, settings_from_env

KEY = "test-key-0123456789abcdef"  # fictional, for tests only
Q = {"question": ACME_REVENUE_Q}


class FakeClock:
    """Time that only moves when the test says so, so the rate limit can be
    tested without real waiting."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def make(retriever, **kwargs):
    llm = FakeLLM()
    app = create_app(retriever=retriever, llm=llm, **kwargs)
    return app, llm


# ------------------------------------------------------------ no key configured

def test_without_a_configured_key_everything_stays_open(retriever):
    app, _ = make(retriever)
    c = TestClient(app)
    assert c.post("/search", json=Q).status_code == 200
    assert c.post("/ask", json=Q).status_code == 200


# ------------------------------------------------------------ key configured

def test_health_stays_open_for_monitoring(retriever):
    app, _ = make(retriever, api_key=KEY)
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["auth_required"] is True


def test_correct_key_is_accepted(retriever):
    app, llm = make(retriever, api_key=KEY)
    c = TestClient(app)
    assert c.post("/search", json=Q, headers={"X-API-Key": KEY}).status_code == 200
    assert c.post("/ask", json=Q, headers={"X-API-Key": KEY}).status_code == 200
    assert llm.calls == 1


@pytest.mark.parametrize("headers", [
    {},                                          # no key at all
    {"X-API-Key": ""},                           # empty key
    {"X-API-Key": "wrong"},                      # wrong key
    {"X-API-Key": KEY[:-1]},                     # one character short
    {"X-API-Key": KEY + "x"},                    # one character extra
    {"X-API-Key": KEY.upper()},                  # wrong case
    {"X-API-Key": " " + KEY},                    # leading space
    {"Authorization": f"Bearer {KEY}"},          # right key, wrong header
    {"X-API-Key": "é".encode() * 10},            # non ASCII bytes (sent raw, as an attacker could)
])
@pytest.mark.parametrize("path", ["/search", "/ask"])
def test_wrong_or_missing_key_is_401_and_costs_nothing(retriever, headers, path):
    app, llm = make(retriever, api_key=KEY)
    r = TestClient(app).post(path, json=Q, headers=headers)
    assert r.status_code == 401
    assert llm.calls == 0
    assert KEY not in r.text  # the real key is never echoed back


# ------------------------------------------------------------ rate limit on /ask

def test_ask_is_limited_per_minute_with_retry_after(retriever):
    clock = FakeClock()
    app, llm = make(retriever, ask_per_minute=3, clock=clock)
    c = TestClient(app)
    assert [c.post("/ask", json=Q).status_code for _ in range(3)] == [200, 200, 200]
    r = c.post("/ask", json=Q)
    assert r.status_code == 429
    assert 1 <= int(r.headers["Retry-After"]) <= 60
    assert llm.calls == 3  # the blocked request did not reach the model


def test_limit_resets_after_a_minute(retriever):
    clock = FakeClock()
    app, _ = make(retriever, ask_per_minute=2, clock=clock)
    c = TestClient(app)
    c.post("/ask", json=Q)
    c.post("/ask", json=Q)
    assert c.post("/ask", json=Q).status_code == 429
    clock.t += 61
    assert c.post("/ask", json=Q).status_code == 200


def test_limit_is_per_client(retriever):
    clock = FakeClock()
    app, _ = make(retriever, ask_per_minute=1, clock=clock)
    alice = TestClient(app, client=("10.0.0.1", 5000))
    bob = TestClient(app, client=("10.0.0.2", 5000))
    assert alice.post("/ask", json=Q).status_code == 200
    assert alice.post("/ask", json=Q).status_code == 429
    assert bob.post("/ask", json=Q).status_code == 200  # Alice's use does not block Bob


def test_search_is_not_rate_limited(retriever):
    app, _ = make(retriever, ask_per_minute=1, clock=FakeClock())
    c = TestClient(app)
    assert [c.post("/search", json=Q).status_code for _ in range(5)] == [200] * 5


def test_rejected_requests_do_not_use_up_the_limit(retriever):
    app, llm = make(retriever, api_key=KEY, ask_per_minute=2, clock=FakeClock())
    c = TestClient(app)
    for _ in range(5):
        c.post("/ask", json=Q)                                              # no key -> 401
        c.post("/ask", json={"question": "  "}, headers={"X-API-Key": KEY})  # bad input -> 422
    ok = {"X-API-Key": KEY}
    assert [c.post("/ask", json=Q, headers=ok).status_code for _ in range(3)] == [200, 200, 429]
    assert llm.calls == 2


# ------------------------------------------------------------ settings from the environment

def test_settings_defaults():
    s = settings_from_env({})
    assert s == {"api_key": None, "ask_per_minute": 10}


def test_settings_read_key_and_limit():
    s = settings_from_env({"FILING_ANALYST_API_KEY": KEY, "FILING_ANALYST_ASK_PER_MINUTE": "25"})
    assert s == {"api_key": KEY, "ask_per_minute": 25}


def test_empty_key_counts_as_unset():
    # Docker and shells often pass an empty value for an unset variable.
    assert settings_from_env({"FILING_ANALYST_API_KEY": "   "})["api_key"] is None


@pytest.mark.parametrize("env, message", [
    ({"FILING_ANALYST_API_KEY": "short"}, "at least 16"),        # weak key
    ({"FILING_ANALYST_ASK_PER_MINUTE": "abc"}, "whole number"),  # not a number
    ({"FILING_ANALYST_ASK_PER_MINUTE": "0"}, "at least 1"),
    ({"FILING_ANALYST_ASK_PER_MINUTE": "-5"}, "at least 1"),
    ({"FILING_ANALYST_ASK_PER_MINUTE": "2.5"}, "whole number"),
])
def test_bad_settings_stop_startup_with_a_clear_message(env, message):
    with pytest.raises(ValueError, match=message):
        settings_from_env(env)
