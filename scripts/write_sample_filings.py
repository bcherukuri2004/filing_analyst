"""Write the three small FICTIONAL 10-K filings used by the tests into a folder.

    python scripts/write_sample_filings.py <folder>

Used by the Docker smoke test, so it needs no download and no SEC access.
The companies (Acme Robotics, Borealis Foods) and their numbers are made up."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

from fixtures import ACME_2022, ACME_2023, BOREALIS_2023  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/write_sample_filings.py <folder>")
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for name, html in (("ACME_2023.htm", ACME_2023), ("ACME_2022.htm", ACME_2022), ("BORE_2023.htm", BOREALIS_2023)):
        with open(os.path.join(out, name), "w") as f:
            f.write(html)
    with open(os.path.join(out, "companies.json"), "w") as f:
        json.dump({"ACME": "Acme Robotics, Inc.", "BORE": "Borealis Foods Corporation"}, f)
    print(f"Wrote 3 sample filings to {out}")


if __name__ == "__main__":
    main()
