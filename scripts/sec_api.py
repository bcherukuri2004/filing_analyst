"""Small helpers for talking to SEC EDGAR. The SEC asks every program to
identify itself, so set SEC_USER_AGENT to your name and email, like:

    export SEC_USER_AGENT="Jane Doe jane@example.com"

and it limits requests to 10 per second (this code stays well under)."""
from __future__ import annotations

import os
import sys
import time

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
PAUSE_SECONDS = 0.25


def session() -> requests.Session:
    agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in agent:
        sys.exit('Set SEC_USER_AGENT to your name and email first, for example:\n  export SEC_USER_AGENT="Jane Doe jane@example.com"')
    s = requests.Session()
    s.headers["User-Agent"] = agent
    return s


def get(s: requests.Session, url: str, retries: int = 4) -> requests.Response:
    for attempt in range(retries):
        time.sleep(PAUSE_SECONDS)
        r = s.get(url, timeout=60)
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    r.raise_for_status()
    return r


def ticker_table(raw: dict) -> dict[str, tuple[int, str]]:
    """company_tickers.json -> {TICKER: (cik, company name)}."""
    return {v["ticker"].upper(): (int(v["cik_str"]), v["title"]) for v in raw.values()}


def latest_10k_filings(submissions: dict, count: int) -> list[dict]:
    """The most recent `count` annual reports from a submissions record:
    [{accession, document, report_date, year}], newest first."""
    recent = submissions["filings"]["recent"]
    out = []
    for i, form in enumerate(recent["form"]):
        if form != "10-K":
            continue
        report_date = recent["reportDate"][i]
        out.append({
            "accession": recent["accessionNumber"][i],
            "document": recent["primaryDocument"][i],
            "report_date": report_date,
            "year": int(report_date[:4]),  # the year the fiscal year ended
        })
    out.sort(key=lambda f: f["report_date"], reverse=True)
    return out[:count]


def document_url(cik: int, filing: dict) -> str:
    return DOC_URL.format(cik=cik, accession=filing["accession"].replace("-", ""), document=filing["document"])
