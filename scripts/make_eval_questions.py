"""Build a question file for `python -m filing_analyst eval` from the SEC's
structured financial data (XBRL). The numbers come from the SEC, not from a
language model, so the answers are objective.

    python scripts/make_eval_questions.py data eval/questions.jsonl

For every filing in the folder it asks for revenue, net income, operating
income and total assets, and records the number (written the ways filings
usually write it) as the gold string a retrieved passage must contain."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sec_api  # noqa: E402

from filing_analyst.ingest import list_filings, load_names  # noqa: E402

# question wording, XBRL tags to try in order, and whether the value is over a period or at a date
METRICS = [
    ("total revenue", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"], "period"),
    ("net income", ["NetIncomeLoss", "ProfitLoss"], "period"),
    ("operating income", ["OperatingIncomeLoss"], "period"),
    ("total assets", ["Assets"], "instant"),
]


def gold_strings(value: float) -> list[str]:
    """How a filing might print this dollar amount: in millions, in thousands,
    or in full. Short numbers are skipped because they match too much text."""
    v = abs(round(value))
    out = []
    for scale in (1_000_000, 1_000, 1):
        s = f"{round(v / scale):,}"
        if len(s.replace(",", "")) >= 5 and s not in out:
            out.append(s)
    return out


def pick_fact(facts: dict, tags: list[str], kind: str, accession: str, report_date: str) -> float | None:
    """Find the value reported in this specific 10-K for the fiscal year that ended on report_date."""
    for tag in tags:
        series = facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {}).get("USD", [])
        for f in series:
            if f.get("accn") != accession or f.get("end") != report_date:
                continue
            if kind == "period":
                if "start" not in f:
                    continue
                days = (date.fromisoformat(f["end"]) - date.fromisoformat(f["start"])).days
                if not 340 <= days <= 380:  # a full year, not a quarter
                    continue
            return float(f["val"])
    return None


def build_questions(facts: dict, ticker: str, company: str, year: int, accession: str, report_date: str) -> list[dict]:
    rows = []
    for label, tags, kind in METRICS:
        value = pick_fact(facts, tags, kind, accession, report_date)
        if value is None:
            continue
        gold = gold_strings(value)
        if not gold:
            continue
        when = f"at the end of fiscal {year}" if kind == "instant" else f"for fiscal {year}"
        rows.append({"question": f"What was {company}'s {label} {when}?", "ticker": ticker, "year": year, "gold": gold})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="folder with the downloaded filings")
    p.add_argument("out", help="question file to write")
    args = p.parse_args()

    s = sec_api.session()
    names = load_names(args.folder)
    table = sec_api.ticker_table(sec_api.get(s, sec_api.TICKERS_URL).json())
    total = 0
    with open(args.out, "w") as out:
        for ticker, year, _ in list_filings(args.folder):
            if ticker not in table:
                continue
            cik, _ = table[ticker]
            subs = sec_api.get(s, sec_api.SUBMISSIONS_URL.format(cik=cik)).json()
            match = [f for f in sec_api.latest_10k_filings(subs, 20) if f["year"] == year]
            if not match:
                continue
            facts = sec_api.get(s, sec_api.FACTS_URL.format(cik=cik)).json()
            short_name = names.get(ticker, ticker)
            for row in build_questions(facts, ticker, short_name, year, match[0]["accession"], match[0]["report_date"]):
                out.write(json.dumps(row) + "\n")
                total += 1
    print(f"Wrote {total} questions to {args.out}")


if __name__ == "__main__":
    main()
