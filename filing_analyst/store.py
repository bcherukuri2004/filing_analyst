"""Storage: passages and their metadata in SQLite (with its built in FTS5
keyword index), and the embedding vectors in one numpy file."""
from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass

import numpy as np

from .chunk import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    company TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    item TEXT NOT NULL,
    section TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, section, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

STOPWORDS = set(
    "a an and are as at be by did do does for from had has have how in is it its of on or that the their this to was "
    "were what when where which who why will with about into than then they them there these those".split()
)


@dataclass
class Hit:
    id: int
    ticker: str
    company: str
    fiscal_year: int
    item: str
    section: str
    kind: str
    text: str
    score: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.company} {self.fiscal_year} 10-K, Item {self.item} {self.section}"


def fts_query(question: str) -> str:
    """Build a safe FTS5 query: every meaningful word, OR'd together. Words are
    quoted so characters like '-' or ':' cannot be read as search syntax."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", question.lower()) if w not in STOPWORDS and len(w) > 1]
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(f'"{w}"')
    return " OR ".join(out)


class Store:
    def __init__(self, directory: str):
        self.dir = directory
        os.makedirs(directory, exist_ok=True)
        # check_same_thread=False lets a web server's worker threads use this
        # connection. Callers that use it from several threads must take turns
        # (the API does this with a lock); sqlite3 connections are not safe to
        # use from two threads at the same moment.
        self.db = sqlite3.connect(os.path.join(directory, "filings.db"), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._emb: np.ndarray | None = None
        self._ids: np.ndarray | None = None

    # ---- writing ----
    def replace_all(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str) -> None:
        assert len(chunks) == len(vectors)
        with self.db:
            self.db.execute("DELETE FROM chunks")
            self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
            for i, c in enumerate(chunks, start=1):
                self.db.execute(
                    "INSERT INTO chunks(id,ticker,company,fiscal_year,item,section,ordinal,kind,text) VALUES(?,?,?,?,?,?,?,?,?)",
                    (i, c.ticker, c.company, c.fiscal_year, c.item, c.section, c.ordinal, c.kind, c.text),
                )
            self.db.execute("INSERT INTO chunks_fts(rowid,text,section) SELECT id,text,section FROM chunks")
            self.db.execute("INSERT OR REPLACE INTO meta VALUES('embedder',?)", (embedder_name,))
        np.save(os.path.join(self.dir, "vectors.npy"), vectors.astype(np.float32))
        self._emb = None

    # ---- reading ----
    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def embedder_name(self) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
        return row[0] if row else None

    def companies(self) -> list[tuple[str, str]]:
        rows = self.db.execute("SELECT DISTINCT ticker, company FROM chunks ORDER BY ticker").fetchall()
        return [(r["ticker"], r["company"]) for r in rows]

    def any_contains(self, ticker: str, year: int, strings: list[str]) -> bool:
        """Does any passage of this filing contain one of the strings?"""
        for text in strings:
            row = self.db.execute(
                "SELECT 1 FROM chunks WHERE ticker=? AND fiscal_year=? AND instr(text, ?) > 0 LIMIT 1", (ticker, year, text)
            ).fetchone()
            if row:
                return True
        return False

    def years(self, ticker: str) -> list[int]:
        rows = self.db.execute("SELECT DISTINCT fiscal_year FROM chunks WHERE ticker=? ORDER BY 1", (ticker,)).fetchall()
        return [r[0] for r in rows]

    def _hit(self, row, score: float) -> Hit:
        return Hit(row["id"], row["ticker"], row["company"], row["fiscal_year"], row["item"], row["section"],
                   row["kind"], row["text"], score)

    def get(self, ids: list[int]) -> list[Hit]:
        if not ids:
            return []
        q = ",".join("?" * len(ids))
        rows = {r["id"]: r for r in self.db.execute(f"SELECT * FROM chunks WHERE id IN ({q})", ids)}
        return [self._hit(rows[i], 0.0) for i in ids if i in rows]

    def keyword_search(self, question: str, k: int, ticker: str | None = None, year: int | None = None) -> list[Hit]:
        q = fts_query(question)
        if not q:
            return []
        sql = ("SELECT c.*, bm25(chunks_fts) AS rank FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
               "WHERE chunks_fts MATCH ?")
        args: list = [q]
        if ticker:
            sql += " AND c.ticker = ?"
            args.append(ticker)
        if year:
            sql += " AND c.fiscal_year = ?"
            args.append(year)
        sql += " ORDER BY rank LIMIT ?"
        args.append(k)
        return [self._hit(r, -r["rank"]) for r in self.db.execute(sql, args)]

    def _load_vectors(self) -> None:
        if self._emb is None:
            self._emb = np.load(os.path.join(self.dir, "vectors.npy"))
            ids = [r[0] for r in self.db.execute("SELECT id FROM chunks ORDER BY id")]
            self._ids = np.array(ids)

    def dense_search(self, qvec: np.ndarray, k: int, ticker: str | None = None, year: int | None = None) -> list[Hit]:
        self._load_vectors()
        scores = self._emb @ qvec
        if ticker or year:
            sql, args = "SELECT id FROM chunks WHERE 1=1", []
            if ticker:
                sql += " AND ticker=?"
                args.append(ticker)
            if year:
                sql += " AND fiscal_year=?"
                args.append(year)
            allowed = {r[0] for r in self.db.execute(sql, args)}
            mask = np.array([i in allowed for i in self._ids])
            scores = np.where(mask, scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        ids = [int(self._ids[i]) for i in top]
        by_id = {h.id: h for h in self.get(ids)}
        return [Hit(**{**by_id[i].__dict__, "score": float(scores[j])}) for j, i in zip(top, ids)]
