"""Tests for the server-wide /ask limit and for grouping IPv6 addresses.

The per-client limit stops one caller from spending too much. It cannot bound
the total: an attacker with many addresses gets the per-client allowance on
each one. IPv6 makes addresses nearly free (one home network usually gets 2^64
of them). So:

  * a global limit caps /ask calls per minute for everyone together, which
    puts a hard ceiling on what the server can spend on the language model;
  * IPv6 addresses in the same /64 network count as one client."""
import pytest
from fastapi.testclient import TestClient
from test_api import ACME_REVENUE_Q, FakeLLM
from test_api_auth import FakeClock

from filing_analyst.api import client_key, create_app, settings_from_env

Q = {"question": ACME_REVENUE_Q}


def make(retriever, **kwargs):
    llm = FakeLLM()
    return create_app(retriever=retriever, llm=llm, **kwargs), llm


def caller(app, host: str) -> TestClient:
    return TestClient(app, client=(host, 5000))


# ------------------------------------------------------------ the global limit

def test_global_limit_caps_everyone_together(retriever):
    app, llm = make(retriever, ask_per_minute=10, ask_global_per_minute=3, clock=FakeClock())
    codes = [caller(app, f"10.0.0.{n}").post("/ask", json=Q).status_code for n in range(1, 7)]
    assert codes == [200, 200, 200, 429, 429, 429]
    assert llm.calls == 3  # the hard ceiling on spending holds


def test_many_addresses_cannot_get_past_the_global_limit(retriever):
    # 500 different IPv6 networks, each well inside its own per-client limit.
    app, llm = make(retriever, ask_per_minute=10, ask_global_per_minute=20, clock=FakeClock())
    for n in range(500):
        caller(app, f"2001:db8:{n:x}::1").post("/ask", json=Q)
    assert llm.calls == 20


def test_global_refusal_says_the_server_is_busy_with_retry_after(retriever):
    app, _ = make(retriever, ask_global_per_minute=1, clock=FakeClock())
    caller(app, "10.0.0.1").post("/ask", json=Q)
    r = caller(app, "10.0.0.2").post("/ask", json=Q)
    assert r.status_code == 429
    assert 1 <= int(r.headers["Retry-After"]) <= 60
    assert "everyone" in r.json()["detail"]  # not blaming this caller for others' use


def test_global_limit_resets_after_a_minute(retriever):
    clock = FakeClock()
    app, _ = make(retriever, ask_global_per_minute=1, clock=clock)
    assert caller(app, "10.0.0.1").post("/ask", json=Q).status_code == 200
    assert caller(app, "10.0.0.2").post("/ask", json=Q).status_code == 429
    clock.t += 61
    assert caller(app, "10.0.0.2").post("/ask", json=Q).status_code == 200


def test_one_client_over_its_own_limit_does_not_use_up_the_global_limit(retriever):
    # Otherwise one noisy caller, refused again and again, could lock everyone out.
    app, llm = make(retriever, ask_per_minute=2, ask_global_per_minute=3, clock=FakeClock())
    noisy = caller(app, "10.0.0.1")
    assert [noisy.post("/ask", json=Q).status_code for _ in range(50)] == [200, 200] + [429] * 48
    assert caller(app, "10.0.0.2").post("/ask", json=Q).status_code == 200
    assert llm.calls == 3


def test_a_global_refusal_does_not_use_up_the_callers_own_limit(retriever):
    # Being turned away because the server is busy must not count against you.
    clock = FakeClock()
    app, _ = make(retriever, ask_per_minute=1, ask_global_per_minute=2, clock=clock)
    assert caller(app, "10.0.0.1").post("/ask", json=Q).status_code == 200   # t=0
    assert caller(app, "10.0.0.2").post("/ask", json=Q).status_code == 200   # t=0, global full
    clock.t += 30
    late = caller(app, "10.0.0.3")
    assert late.post("/ask", json=Q).status_code == 429                      # t=30, busy
    clock.t += 31                                                             # t=61, global free again
    assert late.post("/ask", json=Q).status_code == 200  # its refused try at t=30 did not count


def test_rejected_requests_do_not_count_globally(retriever):
    key = "test-key-0123456789abcdef"  # fictional, for tests only
    app, llm = make(retriever, api_key=key, ask_global_per_minute=2, clock=FakeClock())
    c = caller(app, "10.0.0.1")
    for _ in range(10):
        c.post("/ask", json=Q)                                              # no key -> 401
        c.post("/ask", json={"question": ""}, headers={"X-API-Key": key})   # bad input -> 422
    ok = {"X-API-Key": key}
    assert [c.post("/ask", json=Q, headers=ok).status_code for _ in range(3)] == [200, 200, 429]
    assert llm.calls == 2


def test_search_is_not_limited_by_the_global_cap(retriever):
    app, _ = make(retriever, ask_global_per_minute=1, clock=FakeClock())
    c = caller(app, "10.0.0.1")
    assert [c.post("/search", json=Q).status_code for _ in range(5)] == [200] * 5


# ------------------------------------------------------------ IPv6 grouping

@pytest.mark.parametrize("a, b", [
    ("2001:db8:1:2::1", "2001:db8:1:2:ffff:ffff:ffff:ffff"),  # same /64 network
    ("2001:DB8:1:2::1", "2001:db8:1:2::1"),                    # same address, different case
    ("::ffff:10.0.0.1", "10.0.0.1"),                           # IPv4 written as IPv6
])
def test_same_client(a, b):
    assert client_key(a) == client_key(b)


@pytest.mark.parametrize("a, b", [
    ("2001:db8:1:2::1", "2001:db8:1:3::1"),  # neighbouring /64 networks
    ("10.0.0.1", "10.0.0.2"),                # IPv4 addresses stay separate
])
def test_different_clients(a, b):
    assert client_key(a) != client_key(b)


@pytest.mark.parametrize("host", ["unknown", "", "not-an-ip", "999.1.1.1", "2001:db8::zz", "a" * 5000])
def test_strange_hosts_are_kept_as_they_are_and_do_not_crash(host):
    assert client_key(host) == host


def test_addresses_in_one_ipv6_network_share_one_limit(retriever):
    app, llm = make(retriever, ask_per_minute=2, clock=FakeClock())
    codes = [caller(app, f"2001:db8:1:2::{n:x}").post("/ask", json=Q).status_code for n in range(1, 6)]
    assert codes == [200, 200, 429, 429, 429]
    assert caller(app, "2001:db8:1:3::1").post("/ask", json=Q).status_code == 200  # another network
    assert llm.calls == 3


# ------------------------------------------------------------ settings

def test_global_limit_is_on_by_default():
    # The safe choice: a server started with no settings still has a ceiling.
    assert settings_from_env({})["ask_global_per_minute"] == 60


def test_global_limit_is_read_from_the_environment():
    assert settings_from_env({"FILING_ANALYST_ASK_GLOBAL_PER_MINUTE": " 200 "})["ask_global_per_minute"] == 200


@pytest.mark.parametrize("raw, message", [
    ("abc", "whole number"),
    ("2.5", "whole number"),
    ("", "whole number"),
    ("0", "at least 1"),   # 0 must not mean "no limit"
    ("-1", "at least 1"),
])
def test_bad_global_limit_stops_startup(raw, message):
    with pytest.raises(ValueError, match=f"FILING_ANALYST_ASK_GLOBAL_PER_MINUTE.*{message}"):
        settings_from_env({"FILING_ANALYST_ASK_GLOBAL_PER_MINUTE": raw})
