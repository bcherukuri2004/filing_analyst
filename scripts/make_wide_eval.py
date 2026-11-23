"""Build the wider eval question file from the tagged numbers (inline XBRL)
inside the downloaded 10-K filings. No network needed.

    python scripts/make_wide_eval.py data eval/questions_wide.jsonl

Up to 13 figures per filing (revenue, costs, R&D, profit, tax, cash flow,
capital spending and the main balance sheet lines), each asked in one of three
wordings, with the number exactly as the filing prints it as the answer. See
filing_analyst/xbrl.py for how the right figure is chosen."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filing_analyst.ingest import list_filings, load_names  # noqa: E402
from filing_analyst.xbrl import build_questions, read_facts  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("folder", help="folder with the downloaded filings")
    p.add_argument("out", help="question file to write")
    args = p.parse_args(argv)

    names = load_names(args.folder)
    rows = []
    for ticker, year, path in list_filings(args.folder):
        with open(path, encoding="utf-8", errors="replace") as f:
            facts = read_facts(f.read())
        if not facts:
            print(f"  {os.path.basename(path)}: no tagged figures found (not inline XBRL?), skipped")
            continue
        found = build_questions(facts, ticker, names.get(ticker, ticker), year)
        print(f"  {os.path.basename(path)}: {len(found)} questions")
        rows.extend(found)
    if not rows:
        sys.exit("No questions could be built.")
    with open(args.out, "w") as out:
        for r in rows:
            out.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} questions to {args.out}")


if __name__ == "__main__":
    main()
