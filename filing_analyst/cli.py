"""Command line: ingest, search, ask, eval."""
from __future__ import annotations

import argparse
import os
import sys

from .store import open_store

DEFAULT_INDEX = "index"


def _embedder():
    from .embed import FastEmbedder

    return FastEmbedder()


def _reranker(enabled: bool):
    if not enabled:
        return None
    from .retrieve import CrossEncoderReranker

    return CrossEncoderReranker()


def _show_hits(hits, width: int = 300) -> None:
    for i, h in enumerate(hits, start=1):
        snippet = " ".join(h.text.split())
        print(f"[{i}] {h.label}  (score {h.score:.3f})")
        print(f"    {snippet[:width]}{'...' if len(snippet) > width else ''}")


def cmd_ingest(args) -> None:
    from .ingest import ingest_folder

    store = open_store(args.index)
    n = ingest_folder(args.folder, store, _embedder())
    print(f"Indexed {n} passages into {store.location}")


def cmd_search(args) -> None:
    from .retrieve import Retriever

    store = open_store(args.index)
    needs_dense = args.mode in ("dense", "hybrid")
    r = Retriever(store, _embedder() if needs_dense else None, _reranker(not args.no_rerank))
    hits, filters = r.search(args.question, k=args.k, mode=args.mode, rerank=not args.no_rerank,
                             use_filters=not args.no_filters)
    if filters:
        print(f"(restricted to {filters.tickers or 'all companies'}, year {filters.year or 'any'})")
    _show_hits(hits)


def cmd_ask(args) -> None:
    from .answer import AnthropicLLM, answer_question
    from .retrieve import Retriever

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first. (The search command works without one.)")
    store = open_store(args.index)
    r = Retriever(store, _embedder(), _reranker(not args.no_rerank))
    hits, filters = r.search(args.question, k=args.k, rerank=not args.no_rerank)
    ans = answer_question(args.question, hits, AnthropicLLM())
    print(ans.text)
    if ans.refused:
        return
    print()
    for n in ans.cited:
        h = hits[n - 1]
        print(f"[{n}] {h.label}")
    if ans.problems:
        print("\nWARNING: this answer failed a check, do not rely on it:")
        for p in ans.problems:
            print("  -", p)


def cmd_eval(args) -> None:
    from .evaluate import run_eval

    run_eval(args.questions, open_store(args.index), _embedder(), args.k, with_rerank=not args.no_rerank)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="filing_analyst", description="Ask questions about SEC 10-K filings.")
    p.add_argument("--index", default=DEFAULT_INDEX, help="folder for the search index, or a postgresql:// address (default: index)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="parse and index the filings in a folder")
    s.add_argument("folder")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("search", help="show the best matching passages (no API key needed)")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    s.add_argument("--no-rerank", action="store_true")
    s.add_argument("--no-filters", action="store_true", help="do not read company and year from the question")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("ask", help="answer a question with citations (needs ANTHROPIC_API_KEY)")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--no-rerank", action="store_true")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("eval", help="measure retrieval quality on a question file")
    s.add_argument("questions")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--no-rerank", action="store_true", help="skip the reranker rows")
    s.set_defaults(fn=cmd_eval)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
