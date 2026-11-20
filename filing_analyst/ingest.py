"""Read every filing in a folder, cut it into passages, embed them, and save
the index."""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable

from .chunk import Chunk, chunk_filing
from .embed import Embedder
from .parse import parse_filing
from .store import Store

FILENAME_RE = re.compile(r"^([A-Za-z.]+)_(\d{4})\.html?$")  # AAPL_2023.htm


def list_filings(folder: str) -> list[tuple[str, int, str]]:
    """(ticker, fiscal year, path) for every file named TICKER_YEAR.htm."""
    found = []
    for name in sorted(os.listdir(folder)):
        m = FILENAME_RE.match(name)
        if m:
            found.append((m.group(1).upper(), int(m.group(2)), os.path.join(folder, name)))
    return found


def load_names(folder: str) -> dict[str, str]:
    path = os.path.join(folder, "companies.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def ingest_folder(folder: str, store: Store, embedder: Embedder, log: Callable[[str], None] = print) -> int:
    names = load_names(folder)
    filings = list_filings(folder)
    if not filings:
        raise SystemExit(f"No files named like AAPL_2023.htm found in {folder}")
    all_chunks: list[Chunk] = []
    for ticker, year, path in filings:
        with open(path, encoding="utf-8", errors="replace") as f:
            sections, parsed_year = parse_filing(f.read())
        if not sections:
            log(f"  {os.path.basename(path)}: no sections found, skipped (is it a 10-K?)")
            continue
        if parsed_year and parsed_year != year:
            log(f"  {os.path.basename(path)}: filename says {year} but the filing says {parsed_year}; using {year}")
        chunks = chunk_filing(sections, ticker, names.get(ticker, ticker), year)
        for c in chunks:
            c.ordinal += len(all_chunks)  # keep ordinals unique across filings
        words = sum(len(c.text.split()) for c in chunks)
        log(f"  {os.path.basename(path)}: {len(sections)} sections, {len(chunks)} passages, {words:,} words")
        all_chunks.extend(chunks)
    if not all_chunks:
        raise SystemExit("Nothing to index.")
    log(f"Embedding {len(all_chunks)} passages with {embedder.name} ...")
    vectors = embedder.embed_passages([c.embed_text for c in all_chunks])
    store.replace_all(all_chunks, vectors, embedder.name)
    return len(all_chunks)
