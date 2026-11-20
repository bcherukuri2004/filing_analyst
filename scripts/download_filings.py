"""Download recent 10-K annual reports from SEC EDGAR into a folder.

    export SEC_USER_AGENT="Your Name you@example.com"
    python scripts/download_filings.py data AAPL MSFT NVDA --years 2

Files are saved as TICKER_YEAR.htm, where YEAR is the calendar year in which
the company's fiscal year ended. A companies.json file records each ticker's
company name for the index."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_api  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("tickers", nargs="+")
    p.add_argument("--years", type=int, default=2, help="how many recent annual reports per company (default 2)")
    args = p.parse_args()

    s = sec_api.session()
    os.makedirs(args.folder, exist_ok=True)
    table = sec_api.ticker_table(sec_api.get(s, sec_api.TICKERS_URL).json())

    names_path = os.path.join(args.folder, "companies.json")
    names = json.load(open(names_path)) if os.path.exists(names_path) else {}

    for ticker in (t.upper() for t in args.tickers):
        if ticker not in table:
            print(f"{ticker}: not found in the SEC ticker list, skipped")
            continue
        cik, name = table[ticker]
        subs = sec_api.get(s, sec_api.SUBMISSIONS_URL.format(cik=cik)).json()
        filings = sec_api.latest_10k_filings(subs, args.years)
        if not filings:
            print(f"{ticker}: no recent 10-K found")
            continue
        names[ticker] = name
        for f in filings:
            path = os.path.join(args.folder, f"{ticker}_{f['year']}.htm")
            if os.path.exists(path):
                print(f"{ticker} {f['year']}: already downloaded")
                continue
            r = sec_api.get(s, sec_api.document_url(cik, f))
            with open(path, "wb") as out:
                out.write(r.content)
            print(f"{ticker} {f['year']}: saved {len(r.content) / 1e6:.1f} MB ({f['report_date']})")
    with open(names_path, "w") as out:
        json.dump(names, out, indent=2)
    print(f"Done. Next: python -m filing_analyst ingest {args.folder}")


if __name__ == "__main__":
    main()
