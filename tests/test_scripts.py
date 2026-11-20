"""Tests for the pure logic in the SEC scripts, using hand written JSON shaped
like the SEC's responses (the real site cannot be reached from the test
environment)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import make_eval_questions as mq  # noqa: E402
import sec_api  # noqa: E402

SUBMISSIONS = {"filings": {"recent": {
    "form": ["10-Q", "10-K", "8-K", "10-K", "10-K/A", "10-K"],
    "accessionNumber": ["0001-24-000001", "0001-23-000002", "0001-23-000003", "0001-22-000004", "0001-23-000005", "0001-21-000006"],
    "primaryDocument": ["q.htm", "a23.htm", "e.htm", "a22.htm", "amend.htm", "a21.htm"],
    "reportDate": ["2024-03-31", "2023-09-30", "", "2022-09-24", "2023-09-30", "2021-09-25"],
}}}


def test_ticker_table():
    raw = {"0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple Inc."}}
    assert sec_api.ticker_table(raw) == {"AAPL": (320193, "Apple Inc.")}


def test_latest_10k_filings_picks_annual_reports_newest_first():
    out = sec_api.latest_10k_filings(SUBMISSIONS, 2)
    assert [f["year"] for f in out] == [2023, 2022]  # no 10-Q, 8-K or 10-K/A
    assert out[0]["document"] == "a23.htm"


def test_document_url():
    f = sec_api.latest_10k_filings(SUBMISSIONS, 1)[0]
    assert sec_api.document_url(320193, f) == "https://www.sec.gov/Archives/edgar/data/320193/000123000002/a23.htm"


def test_session_requires_contact(monkeypatch):
    import pytest

    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(SystemExit):
        sec_api.session()
    monkeypatch.setenv("SEC_USER_AGENT", "Jane Doe jane@example.com")
    assert sec_api.session().headers["User-Agent"] == "Jane Doe jane@example.com"


def test_gold_strings():
    assert mq.gold_strings(383285000000) == ["383,285", "383,285,000", "383,285,000,000"]
    assert mq.gold_strings(-5000000) == ["5,000,000"]  # a loss is matched by its absolute value; "5,000" is too short
    assert mq.gold_strings(1234) == []  # too short to match reliably


def fact(val, end, start=None, accn="A-1"):
    d = {"val": val, "end": end, "accn": accn}
    if start:
        d["start"] = start
    return d


def test_pick_fact_uses_the_right_filing_and_a_full_year():
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        fact(90, "2023-09-30", "2023-07-02"),                 # one quarter: wrong length
        fact(100, "2022-09-24", "2021-09-26", accn="A-0"),    # prior year, other filing
        fact(380, "2023-09-30", "2022-09-25"),                # the one we want
        fact(111, "2023-09-30", "2022-09-25", accn="A-2"),    # same period in a different filing
    ]}}}}}
    assert mq.pick_fact(facts, ["Revenues"], "period", "A-1", "2023-09-30") == 380
    assert mq.pick_fact(facts, ["Nope"], "period", "A-1", "2023-09-30") is None
    facts["facts"]["us-gaap"]["Assets"] = {"units": {"USD": [fact(5000000, "2023-09-30")]}}
    assert mq.pick_fact(facts, ["Assets"], "instant", "A-1", "2023-09-30") == 5000000


def test_build_questions():
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [fact(383285000000, "2023-09-30", "2022-10-01")]}},
        "Assets": {"units": {"USD": [fact(352583000000, "2023-09-30")]}},
    }}}
    rows = mq.build_questions(facts, "AAPL", "Apple", 2023, "A-1", "2023-09-30")
    assert [r["question"] for r in rows] == [
        "What was Apple's total revenue for fiscal 2023?",
        "What was Apple's total assets at the end of fiscal 2023?",
    ]
    assert rows[0]["gold"][0] == "383,285" and rows[0]["ticker"] == "AAPL" and rows[0]["year"] == 2023
