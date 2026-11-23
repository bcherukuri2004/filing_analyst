"""Storage. Two stores with the same methods:

    Store    SQLite (with its built in FTS5 keyword index) plus the vectors in
             one numpy file, in a folder. The default: nothing to set up.
    PgStore  Postgres, with its built in full text search for keywords and the
             pgvector extension for vectors. For a shared database server.

open_store() picks one from a location: a postgresql:// address means
Postgres, anything else is a folder for SQLite."""
from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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
    return " OR ".join(f'"{w}"' for w in query_words(question))


def query_words(question: str) -> list[str]:
    """The meaningful words of a question, lower case, each once, in order.
    Only letters and digits, so no search syntax can get through."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", question.lower()) if w not in STOPWORDS and len(w) > 1]
    return list(dict.fromkeys(words))


def check_input(chunks: list[Chunk], vectors: np.ndarray) -> None:
    """Refuse an index that would be broken, before anything is deleted."""
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} passages but {len(vectors)} vectors")
    if not np.isfinite(vectors).all():
        raise ValueError("some vectors contain NaN or infinite values")


def _clean(text: str) -> str:
    return text.replace("\x00", "")  # Postgres cannot store NUL, so neither store keeps it


def is_postgres(location: str) -> bool:
    return location.startswith(("postgresql://", "postgres://"))


def describe_location(location: str) -> str:
    """The location with any password replaced by ***, safe for logs."""
    if not is_postgres(location):
        return location
    parts = urlsplit(location)
    netloc = parts.netloc
    if parts.password is not None:
        user_info, host = netloc.rsplit("@", 1)
        netloc = f"{user_info.split(':', 1)[0]}:***@{host}"
    query = urlencode([(k, "***" if k == "password" else v) for k, v in parse_qsl(parts.query)], safe="*")
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def open_store(location: str):
    """A PgStore for a postgresql:// address, otherwise a SQLite Store in that folder."""
    return PgStore(location) if is_postgres(location) else Store(location)


class Store:
    kind = "sqlite"

    def __init__(self, directory: str):
        self.dir = directory
        self.location = directory
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
        check_input(chunks, vectors)
        with self.db:
            self.db.execute("DELETE FROM chunks")
            self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
            for i, c in enumerate(chunks, start=1):
                self.db.execute(
                    "INSERT INTO chunks(id,ticker,company,fiscal_year,item,section,ordinal,kind,text) VALUES(?,?,?,?,?,?,?,?,?)",
                    (i, c.ticker, c.company, c.fiscal_year, c.item, _clean(c.section), c.ordinal, c.kind, _clean(c.text)),
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
        # Highest score first; equal scores in id order, so results never depend on chance
        # (and match Postgres, which sorts the same way). A full sort of a few thousand
        # scores takes well under a millisecond.
        top = np.lexsort((self._ids, -scores))[:k]
        ids = [int(self._ids[i]) for i in top]
        by_id = {h.id: h for h in self.get(ids)}
        return [Hit(**{**by_id[i].__dict__, "score": float(scores[j])}) for j, i in zip(top, ids)]


class PgStore:
    """The same store in Postgres. Keyword search uses Postgres full text
    search (ranked with ts_rank_cd, which is not BM25, so scores differ from
    SQLite). Vector search uses pgvector with an HNSW index, and company and
    year filters run inside the same query.

    One connection, used by one caller at a time (the API takes turns with a
    lock, as it does for SQLite)."""

    kind = "postgres"

    # How the HNSW vector index is searched:
    # - iterative_scan: when the index's first batch of candidates runs out
    #   (pgvector looks at 40 by default, the retriever asks for 50) or a filter
    #   removes most of them, keep scanning. Without it a search can quietly
    #   return fewer rows than asked for. A test forces the index on and checks.
    # - ef_search: candidates per batch. Higher is closer to exact and a little
    #   slower; it affects accuracy, not how many rows come back.
    EF_SEARCH = 200

    def __init__(self, url: str):
        import psycopg
        from psycopg.rows import dict_row

        self.location = describe_location(url)
        try:
            self.db = psycopg.connect(url, autocommit=True, row_factory=dict_row)
        except psycopg.Error as e:
            # The driver's message can repeat parts of the address; never the password.
            message = f"Could not connect to Postgres at {self.location}: {str(e).strip()}"
            password = urlsplit(url).password or dict(parse_qsl(urlsplit(url).query)).get("password")
            if password:
                message = message.replace(password, "***")
            raise ConnectionError(message) from None
        self.db.execute("CREATE EXTENSION IF NOT EXISTS vector")
        from pgvector.psycopg import register_vector

        register_vector(self.db)
        self.db.execute("SET hnsw.ef_search = %d" % self.EF_SEARCH)
        self.db.execute("SET hnsw.iterative_scan = relaxed_order")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self.db.execute(self._chunks_table(None).replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS"))

    @staticmethod
    def _chunks_table(dim: int | None) -> str:
        vector = f"vector({dim})" if dim else "vector"
        return f"""CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            company TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            item TEXT NOT NULL,
            section TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding {vector} NOT NULL,
            tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text || ' ' || section)) STORED
        )"""

    # ---- writing ----
    def replace_all(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str) -> None:
        check_input(chunks, vectors)
        vectors = vectors.astype(np.float32)
        dim = vectors.shape[1] if vectors.ndim == 2 and len(vectors) else None
        # One transaction: if anything fails, the old index is left exactly as it was.
        with self.db.transaction():
            self.db.execute("DROP TABLE IF EXISTS chunks")
            self.db.execute(self._chunks_table(dim))  # the vector size can change between indexes
            with self.db.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks(id,ticker,company,fiscal_year,item,section,ordinal,kind,text,embedding) "
                    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [(i, c.ticker, c.company, c.fiscal_year, c.item, _clean(c.section), c.ordinal, c.kind,
                      _clean(c.text), v) for i, (c, v) in enumerate(zip(chunks, vectors), start=1)],
                )
            # Indexes are built after loading, which is much faster than updating them row by row.
            self.db.execute("CREATE INDEX chunks_tsv ON chunks USING gin (tsv)")
            self.db.execute("CREATE INDEX chunks_filter ON chunks (ticker, fiscal_year)")
            if dim:
                self.db.execute("CREATE INDEX chunks_embedding ON chunks USING hnsw (embedding vector_ip_ops)")
            self.db.execute("INSERT INTO meta VALUES('embedder', %s) ON CONFLICT (key) DO UPDATE SET value = "
                            "EXCLUDED.value", (embedder_name,))
        self.db.execute("ANALYZE chunks")  # fresh statistics so the planner picks the indexes

    # ---- reading ----
    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]

    def embedder_name(self) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
        return row["value"] if row else None

    def companies(self) -> list[tuple[str, str]]:
        rows = self.db.execute("SELECT DISTINCT ticker, company FROM chunks ORDER BY ticker").fetchall()
        return [(r["ticker"], r["company"]) for r in rows]

    def years(self, ticker: str) -> list[int]:
        rows = self.db.execute("SELECT DISTINCT fiscal_year FROM chunks WHERE ticker = %s ORDER BY 1", (ticker,))
        return [r["fiscal_year"] for r in rows]

    def any_contains(self, ticker: str, year: int, strings: list[str]) -> bool:
        if not strings:
            return False
        row = self.db.execute(
            "SELECT 1 FROM chunks WHERE ticker = %s AND fiscal_year = %s "
            "AND EXISTS (SELECT 1 FROM unnest(%s::text[]) s WHERE strpos(text, s) > 0) LIMIT 1",
            (ticker, year, [_clean(t) for t in strings]),
        ).fetchone()
        return row is not None

    _COLUMNS = "id, ticker, company, fiscal_year, item, section, kind, text"

    @staticmethod
    def _hit(row, score: float) -> Hit:
        return Hit(row["id"], row["ticker"], row["company"], row["fiscal_year"], row["item"], row["section"],
                   row["kind"], row["text"], score)

    def get(self, ids: list[int]) -> list[Hit]:
        if not ids:
            return []
        rows = {r["id"]: r for r in self.db.execute(
            f"SELECT {self._COLUMNS} FROM chunks WHERE id = ANY(%s)", (list(ids),))}
        return [self._hit(rows[i], 0.0) for i in ids if i in rows]

    @staticmethod
    def _filters(ticker: str | None, year: int | None) -> tuple[str, list]:
        sql, args = "", []
        if ticker:
            sql += " AND ticker = %s"
            args.append(ticker)
        if year:
            sql += " AND fiscal_year = %s"
            args.append(year)
        return sql, args

    def keyword_search(self, question: str, k: int, ticker: str | None = None, year: int | None = None) -> list[Hit]:
        words = query_words(question)
        if not words:
            return []
        where, args = self._filters(ticker, year)
        # Words are letters and digits only, joined with | (OR), and passed as a
        # parameter, so a question can never change the query itself.
        rows = self.db.execute(
            f"SELECT {self._COLUMNS}, ts_rank_cd(tsv, q) AS rank FROM chunks, to_tsquery('english', %s) q "
            f"WHERE tsv @@ q{where} ORDER BY rank DESC, id LIMIT %s",
            [" | ".join(words), *args, k],
        )
        return [self._hit(r, float(r["rank"])) for r in rows]

    def dense_search(self, qvec: np.ndarray, k: int, ticker: str | None = None, year: int | None = None) -> list[Hit]:
        if k <= 0:
            return []
        where, args = self._filters(ticker, year)
        q = np.asarray(qvec, dtype=np.float32)
        # <#> is pgvector's negative inner product: smallest first = most similar.
        # With relaxed_order the index may hand back rows slightly out of order,
        # so the outer query sorts the final list (as the pgvector docs advise;
        # in practice the rows usually arrive sorted already).
        rows = self.db.execute(
            f"WITH nearest AS MATERIALIZED (SELECT {self._COLUMNS}, embedding <#> %s AS distance FROM chunks "
            f"WHERE true{where} ORDER BY distance LIMIT %s) SELECT * FROM nearest ORDER BY distance, id",
            [q, *args, k],
        )
        return [self._hit(r, -float(r["distance"])) for r in rows]
