"""Cut sections into passages small enough to retrieve precisely, while
keeping tables together and remembering where each passage came from."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .parse import Block, Section

MAX_WORDS = 300
MIN_WORDS = 60
OVERLAP_MAX_WORDS = 80  # a short last paragraph is repeated at the start of the next chunk
TABLE_ROWS_PER_CHUNK = 30

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Chunk:
    ticker: str
    company: str
    fiscal_year: int
    item: str
    section: str  # section title, e.g. "Risk Factors"
    ordinal: int  # position within the filing
    kind: str  # "text" or "table"
    text: str

    @property
    def header(self) -> str:
        """A one line label that is added to the text before embedding, so a
        passage from Apple's risk factors does not look the same as one from
        Microsoft's."""
        return f"{self.company} ({self.ticker}) fiscal {self.fiscal_year} Form 10-K, Item {self.item} {self.section}"

    @property
    def embed_text(self) -> str:
        return f"{self.header}\n{self.text}"


def _split_long(text: str) -> list[str]:
    """Break one very long paragraph at sentence boundaries."""
    sentences = _SENT_RE.split(text)
    parts, cur, count = [], [], 0
    for s in sentences:
        w = len(s.split())
        if cur and count + w > MAX_WORDS:
            parts.append(" ".join(cur))
            cur, count = [], 0
        cur.append(s)
        count += w
    if cur:
        parts.append(" ".join(cur))
    return parts


def _table_chunks(text: str, caption: str | None) -> list[str]:
    rows = text.split("\n")
    out = []
    for i in range(0, len(rows), TABLE_ROWS_PER_CHUNK):
        body = "\n".join(rows[i:i + TABLE_ROWS_PER_CHUNK])
        out.append(f"{caption}\n{body}" if caption else body)
    return out


def chunk_section(sec: Section) -> list[tuple[str, str]]:
    """Returns (kind, text) pairs for one section."""
    out: list[tuple[str, str]] = []
    cur: list[str] = []
    count = 0
    last_text = ""  # most recent paragraph, used as a caption for a table

    def flush(keep_overlap: bool) -> None:
        nonlocal cur, count
        if not cur:
            return
        out.append(("text", "\n".join(cur)))
        tail = cur[-1]
        if keep_overlap and len(tail.split()) <= OVERLAP_MAX_WORDS and len(cur) > 1:
            cur, count = [tail], len(tail.split())
        else:
            cur, count = [], 0

    for b in sec.blocks:
        if b.kind == "table":
            flush(False)
            caption = last_text if last_text and len(last_text.split()) <= 40 else None
            for t in _table_chunks(b.text, caption):
                out.append(("table", t))
            continue
        pieces = _split_long(b.text) if len(b.text.split()) > MAX_WORDS else [b.text]
        for piece in pieces:
            w = len(piece.split())
            if cur and count + w > MAX_WORDS and count >= MIN_WORDS:
                flush(True)
            cur.append(piece)
            count += w
        last_text = b.text
    # A leftover that is only the overlap from the previous chunk adds nothing.
    if cur and not (out and out[-1][0] == "text" and out[-1][1].endswith(cur[-1]) and len(cur) == 1):
        out.append(("text", "\n".join(cur)))
    return out


def chunk_filing(sections: list[Section], ticker: str, company: str, fiscal_year: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in sections:
        for kind, text in chunk_section(sec):
            chunks.append(Chunk(ticker, company, fiscal_year, sec.item, sec.title, len(chunks), kind, text))
    return chunks
