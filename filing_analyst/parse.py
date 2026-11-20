"""Turn a 10-K filing (the HTML file from SEC EDGAR) into clean text blocks
grouped by section ("Item 1A. Risk Factors", "Item 7. MD&A", and so on)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

# Items that exist in a 10-K. Anything else that starts with "Item" is ignored.
VALID_ITEMS = {
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8",
    "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16",
}
ITEM_RE = re.compile(r"^\s*item\s*(\d{1,2}[abc]?)\s*[\.\:\-–—]?\s*(.*)$", re.I)
BLOCK_TAGS = {"p", "div", "li", "pre", "h1", "h2", "h3", "h4", "h5", "h6", "center", "section", "blockquote"}
MIN_SECTION_WORDS = 60  # shorter "sections" are table of contents lines, not real sections
MAX_HEADING_CHARS = 200


@dataclass
class Block:
    kind: str  # "text" or "table"
    text: str


@dataclass
class Section:
    item: str  # e.g. "1A"
    title: str  # e.g. "Risk Factors"
    blocks: list[Block] = field(default_factory=list)

    @property
    def words(self) -> int:
        return sum(len(b.text.split()) for b in self.blocks)


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("​", "")
    return re.sub(r"\s+", " ", text).strip()


def _table_to_text(table) -> str:
    """One line per row, cells joined with ' | '. EDGAR tables put '$', ')'
    and '%' in cells of their own, so those are glued back onto the number."""
    lines = []
    for tr in table.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            t = _clean(td.get_text(""))
            if not t:
                continue
            if t in (")", "%", ")%") and cells:
                cells[-1] += t
            elif t == "$":
                continue
            else:
                cells.append(t)
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _is_hidden(tag) -> bool:
    style = (tag.get("style") or "").replace(" ", "").lower() if hasattr(tag, "get") else ""
    return "display:none" in style


def html_to_blocks(html: str) -> list[Block]:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "head", "ix:header"]):
        t.decompose()
    for t in soup.find_all(style=True):
        # A tag inside a hidden tag that was already removed is gone too.
        if t.decomposed:
            continue
        if _is_hidden(t):
            t.decompose()

    # Replace each table by a <pre> holding its text, so the leaf scan below
    # sees a table as a single block.
    for table in soup.find_all("table"):
        text = _table_to_text(table)
        if text.strip():
            pre = soup.new_tag("pre")
            pre["data-table"] = "1"
            pre.string = text
            table.replace_with(pre)
        else:
            table.decompose()

    blocks_tags = soup.find_all(BLOCK_TAGS)
    non_leaf: set[int] = set()
    for t in blocks_tags:
        for p in t.parents:
            if p.name in BLOCK_TAGS:
                if id(p) in non_leaf:
                    break
                non_leaf.add(id(p))

    out: list[Block] = []
    for t in blocks_tags:
        if id(t) in non_leaf:
            continue
        if t.name == "pre" and t.get("data-table"):
            text = t.get_text("\n").strip()
            if text:
                out.append(Block("table", text))
            continue
        text = _clean(t.get_text(""))
        if text:
            out.append(Block("text", text))
    if not out:  # plain text filing with no block tags at all
        for line in soup.get_text("\n").splitlines():
            line = _clean(line)
            if line:
                out.append(Block("text", line))
    return out


def _heading_text(b: Block) -> str | None:
    """The text a block would have as a heading. Some filings lay each heading
    out as a one row table ("Item 1A. | Risk Factors"); its cells are joined
    back into one line. Multi row tables are never headings."""
    if b.kind == "text":
        return b.text
    if "\n" in b.text:
        return None
    return b.text.replace(" | ", " ")


def split_sections(blocks: list[Block]) -> list[Section]:
    """Group blocks under their "Item N." headings, skipping the table of
    contents (whose entries are headings with almost nothing after them)."""
    cands: list[tuple[int, str, str]] = []
    for i, b in enumerate(blocks):
        heading = _heading_text(b)
        if heading is None or len(heading) > MAX_HEADING_CHARS:
            continue
        m = ITEM_RE.match(heading)
        if not m:
            continue
        item = m.group(1).upper()
        if item not in VALID_ITEMS:
            continue
        title = m.group(2).strip(" .:-–—")
        cands.append((i, item, title))

    pieces: list[Section] = []
    for n, (start, item, title) in enumerate(cands):
        end = cands[n + 1][0] if n + 1 < len(cands) else len(blocks)
        body = blocks[start + 1:end]
        # Some filings repeat "Item 1A" at the top of every page. Back to back
        # pieces with the same Item are one section; keep the first real title.
        if pieces and pieces[-1].item == item:
            pieces[-1].blocks.extend(body)
            pieces[-1].title = pieces[-1].title or title
        else:
            pieces.append(Section(item=item, title=title, blocks=list(body)))
    return [s for s in pieces if s.words >= MIN_SECTION_WORDS]


_FY_RE = re.compile(r"for the (?:fiscal )?year ended\s+([A-Za-z]+\s+\d{1,2}),?\s+(\d{4})", re.I)


def find_fiscal_year(blocks: list[Block]) -> int | None:
    for b in blocks[:150]:
        m = _FY_RE.search(b.text)
        if m:
            return int(m.group(2))
    return None


def parse_filing(html: str) -> tuple[list[Section], int | None]:
    blocks = html_to_blocks(html)
    return split_sections(blocks), find_fiscal_year(blocks)
