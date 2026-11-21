"""Tests for the rate limiter's memory bound.

The limiter remembers recent calls per client address. An attacker with many
addresses (cheap with IPv6) must not be able to make that table grow without
limit, or make every request slower by filling it."""
import time

import pytest
from fastapi import HTTPException

from filing_analyst.api import RateLimiter


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def ip(n: int) -> str:
    return f"2001:db8::{n:x}"  # documentation IPv6 range, never a real host


def test_table_never_grows_past_the_cap_under_a_flood():
    lim = RateLimiter(limit=5, clock=FakeClock(), max_clients=1000)
    for n in range(50_000):
        lim.check(ip(n))
        assert len(lim.calls) <= 1000
    assert len(lim.calls) == 1000


def test_the_least_recently_seen_client_is_dropped_first():
    clock = FakeClock()
    lim = RateLimiter(limit=5, clock=clock, max_clients=3)
    lim.check("old")
    lim.check("middle")
    lim.check("recent")
    lim.check("old")          # "old" is used again, so it is now the most recent
    lim.check("newcomer")     # table is full: the least recently seen ("middle") goes
    assert set(lim.calls) == {"recent", "old", "newcomer"}


def test_limits_still_work_with_the_cap():
    lim = RateLimiter(limit=2, clock=FakeClock(), max_clients=10)
    lim.check("alice")
    lim.check("alice")
    with pytest.raises(HTTPException) as e:
        lim.check("alice")
    assert e.value.status_code == 429


def test_a_client_that_keeps_trying_stays_blocked_during_a_flood():
    # A blocked client who keeps calling stays "recently seen" (even its
    # refused calls count as activity), so a flood of other addresses cannot
    # push it out of the table and reset its limit.
    lim = RateLimiter(limit=3, clock=FakeClock(), max_clients=100)
    for _ in range(3):
        lim.check("alice")
    refused = 0
    for n in range(5_000):
        lim.check(ip(n))
        if n % 50 == 0:
            with pytest.raises(HTTPException):
                lim.check("alice")
            refused += 1
    assert refused == 100


def test_known_trade_off_a_dropped_client_starts_again():
    # Documented in the README: if a flood pushes a quiet client out of the
    # table, its count is forgotten. Bounded memory is worth that.
    lim = RateLimiter(limit=1, clock=FakeClock(), max_clients=2)
    lim.check("alice")
    lim.check(ip(1))
    lim.check(ip(2))  # alice was least recently seen, so she is dropped
    lim.check("alice")  # allowed again


def test_request_cost_does_not_grow_with_a_full_table():
    # With the table full of active clients, each call must still be cheap.
    # (The old code scanned every entry on every call once it passed 10,000.)
    def time_calls(table_size: int) -> float:
        lim = RateLimiter(limit=5, clock=FakeClock(), max_clients=table_size)
        for n in range(table_size):
            lim.check(ip(n))  # fill the table with clients active "now"
        start = time.perf_counter()
        for n in range(2_000):
            lim.check(ip(table_size + n))
        return time.perf_counter() - start

    small, large = time_calls(100), time_calls(50_000)
    assert large < small * 20 + 0.05, f"2,000 calls took {large:.3f}s with a full table vs {small:.3f}s"


@pytest.mark.parametrize("bad", [0, -1])
def test_cap_must_be_positive(bad):
    with pytest.raises(ValueError):
        RateLimiter(limit=5, clock=FakeClock(), max_clients=bad)
