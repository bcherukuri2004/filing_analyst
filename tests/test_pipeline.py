import json

import numpy as np
import pytest
from fixtures import ACME_2023

from filing_analyst.answer import NOT_FOUND, answer_question, numbers_in, verify
from filing_analyst.chunk import MAX_WORDS, chunk_filing
from filing_analyst.embed import HashingEmbedder
from filing_analyst.evaluate import Question, evaluate, is_relevant, load_questions, score_ranking
from filing_analyst.parse import find_fiscal_year, html_to_blocks, parse_filing, split_sections
from filing_analyst.retrieve import Filters, company_aliases, detect_filters, rrf
from filing_analyst.store import Hit, fts_query


def hit(i, text="x", ticker="ACME", year=2023):
    return Hit(i, ticker, "Acme", year, "7", "MD&A", "text", text)


# ------------------------------------------------------------ parsing

def test_sections_found_and_table_of_contents_skipped():
    sections, fy = parse_filing(ACME_2023)
    assert fy == 2023
    assert [s.item for s in sections] == ["1", "1A", "7"]
    assert sections[0].title == "Business"
    assert sections[2].title.startswith("Management")
    # The table of contents rows ("Item 1. | Business | 3") must not leak into the body.
    body = " ".join(b.text for s in sections for b in s.blocks)
    assert "Business | 3" not in body


def test_hidden_xbrl_and_scripts_removed():
    blocks = html_to_blocks(ACME_2023)
    assert not any("HIDDEN FACT" in b.text or "color:red" in b.text for b in blocks)


def test_table_rows_are_cleaned():
    blocks = html_to_blocks(ACME_2023)
    tables = [b for b in blocks if b.kind == "table" and "Total net revenue" in b.text]
    assert tables, "results table was lost"
    rows = tables[0].text.split("\n")
    assert "Total net revenue | 1,250,400 | 1,125,360" in rows  # the lone "$" cells are gone
    assert "Net income | 88,100 | (44,050)" in rows  # ")" is glued to its number
    assert "Gross margin | 41.5% | 39.2%" in rows  # "%" is glued to its number


def test_words_inside_split_spans_stay_whole():
    blocks = html_to_blocks("<div><span>Rev</span><span>enue grew</span></div>")
    assert blocks[0].text == "Revenue grew"


def test_fiscal_year_missing_is_none():
    assert find_fiscal_year(html_to_blocks("<p>nothing here</p>")) is None


def test_plain_text_without_block_tags():
    blocks = html_to_blocks("Item 1. Business\nWe make things.\n")
    assert [b.text for b in blocks] == ["Item 1. Business", "We make things."]


def test_hidden_cell_with_styled_children_does_not_crash():
    # Seen in a real filing: a hidden table cell whose children also carry a
    # style. Removing the cell destroys the children, which must then be skipped.
    html = ('<table><tr><td style="display:none"><span style="font-weight:bold">SECRET</span>'
            '<span style="color:red">ALSO SECRET</span></td><td>Visible cell</td></tr></table>'
            '<p style="margin:0">Visible paragraph</p>')
    blocks = html_to_blocks(html)
    text = " ".join(b.text for b in blocks)
    assert "Visible cell" in text and "Visible paragraph" in text
    assert "SECRET" not in text


def test_headings_laid_out_as_one_row_tables():
    # Seen in a real filing: each heading is a small table, "Item 1A. | Risk Factors".
    body = "<p>" + "word " * 80 + "</p>"
    html = ("<table><tr><td>Item 1.</td><td>Business</td><td>3</td></tr>"
            "<tr><td>Item 1A.</td><td>Risk Factors</td><td>9</td></tr></table>"  # table of contents
            "<table><tr><td>Item 1.</td><td>Business</td></tr></table>" + body +
            "<table><tr><td>Item 1A.</td><td>Risk Factors</td></tr></table>" + body +
            "<table><tr><td>Revenue</td><td>100</td></tr><tr><td>Item 7.</td><td>see below</td></tr></table>")
    sections = split_sections(html_to_blocks(html))
    assert [(s.item, s.title) for s in sections] == [("1", "Business"), ("1A", "Risk Factors")]
    # A multi row data table that merely mentions an Item stays in the body.
    assert any(b.kind == "table" and "Revenue | 100" in b.text for b in sections[1].blocks)


def test_repeated_page_headers_do_not_split_sections():
    # Seen in a real filing: every page starts with "PART I" / "Item 1A", so the
    # same Item shows up again and again with no title. It is one section.
    page = "<p>" + "word " * 80 + "</p>"
    html = ("<p>Item 1. Business</p>" + page +
            "<p>PART I</p><p>Item 1</p>" + page +
            "<p>Item 1A</p><p>ITEM 1A. RISK FACTORS</p>" + page +
            "<p>PART I</p><p>Item 1A</p>" + page +
            "<p>PART I</p><p>Item 1A</p>" + page)
    sections = split_sections(html_to_blocks(html))
    assert [(s.item, s.title) for s in sections] == [("1", "Business"), ("1A", "RISK FACTORS")]
    assert sections[0].words >= 160 and sections[1].words >= 240  # no text lost in the merge


def test_cross_references_are_not_headings():
    html = ("<p>Item 1A. Risk Factors</p><p>" + "word " * 80 + "see Item 7. for details</p>"
            "<p>As described in Item 1A. Risk Factors above, we face risks.</p>")
    sections = split_sections(html_to_blocks(html))
    assert [s.item for s in sections] == ["1A"]


# ------------------------------------------------------------ chunking

def test_chunks_are_bounded_and_labelled():
    sections, fy = parse_filing(ACME_2023)
    chunks = chunk_filing(sections, "ACME", "Acme Robotics, Inc.", fy)
    assert chunks
    for c in chunks:
        assert c.text.strip()
        assert len(c.text.split()) <= MAX_WORDS * 1.5
        assert "Acme Robotics" in c.embed_text and "Item" in c.embed_text
    assert any(c.kind == "table" and "Total net revenue" in c.text for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_long_paragraph_is_split_at_sentences():
    from filing_analyst.chunk import chunk_section
    from filing_analyst.parse import Block, Section

    para = " ".join(f"Sentence number {i} says something useful." for i in range(400))
    out = chunk_section(Section("7", "MD&A", [Block("text", para)]))
    assert len(out) > 1
    assert all(len(t.split()) <= MAX_WORDS + 10 for _, t in out)
    assert all(t.rstrip().endswith(".") for _, t in out)


def test_large_table_is_split_with_caption():
    from filing_analyst.chunk import chunk_section
    from filing_analyst.parse import Block, Section

    rows = "\n".join(f"Row {i} | {i * 10}" for i in range(75))
    out = chunk_section(Section("8", "Financials", [Block("text", "Selected data (in thousands)."), Block("table", rows)]))
    tables = [t for k, t in out if k == "table"]
    assert len(tables) == 3
    assert all(t.startswith("Selected data") for t in tables)


# ------------------------------------------------------------ store and search

def test_ingest_counts(store):
    assert store.count() > 10
    assert store.companies() == [("ACME", "Acme Robotics, Inc."), ("BORE", "Borealis Foods Corporation")]
    assert store.years("ACME") == [2022, 2023]
    assert store.embedder_name().startswith("hashing")


def test_fts_query_is_safe():
    assert fts_query("What was Acme's revenue?") == '"acme" OR "revenue"'
    assert fts_query("") == ""
    assert fts_query("the of and") == ""
    # Characters that mean something in FTS syntax must not break the query.
    for nasty in ['revenue AND (2023', 'NEAR("x"', "a-b:c*d", '"quoted', "col:val OR", "'; DROP TABLE chunks; --"]:
        q = fts_query(nasty)
        assert '"' in q or q == ""


def test_keyword_search_finds_revenue_and_respects_filters(store):
    hits = store.keyword_search("total net revenue", 5, ticker="ACME", year=2023)
    assert hits and all(h.ticker == "ACME" and h.fiscal_year == 2023 for h in hits)
    assert any("1,250,400" in h.text for h in hits)
    assert store.keyword_search("???", 5) == []
    assert store.keyword_search("revenue", 5, ticker="NOPE") == []


def test_dense_search_and_filters(store):
    q = HashingEmbedder().embed_query("lithium battery supply shortage")
    hits = store.dense_search(q, 3)
    assert len(hits) == 3 and hits[0].score >= hits[1].score >= hits[2].score
    assert "lithium" in hits[0].text.lower()
    only_bore = store.dense_search(q, 50, ticker="BORE")
    assert only_bore and all(h.ticker == "BORE" for h in only_bore)
    assert store.dense_search(q, 5, ticker="NOPE") == []


def test_hit_ids_match_vectors(store):
    """The row of the vector file must be the same passage as the row in the
    database, otherwise vector search returns the wrong text."""
    emb = HashingEmbedder()
    for h in store.get([1, 5, 9]):
        qv = emb.embed_passages([f"{h.company} ({h.ticker}) fiscal {h.fiscal_year} Form 10-K, Item {h.item} {h.section}\n{h.text}"])[0]
        top = store.dense_search(qv, 1, ticker=h.ticker, year=h.fiscal_year)[0]
        assert top.id == h.id


# ------------------------------------------------------------ retrieval

def test_rrf_merges_ranks():
    a = [hit(1), hit(2), hit(3)]
    b = [hit(3), hit(1), hit(4)]
    merged = rrf([a, b])
    assert [h.id for h in merged] == [1, 3, 2, 4]  # 1 and 3 are in both lists
    assert merged[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert rrf([]) == []


def test_company_aliases():
    assert "apple" in company_aliases("AAPL", "Apple Inc.")
    assert "meta platforms" in company_aliases("META", "Meta Platforms, Inc.")
    assert "meta" in company_aliases("META", "Meta Platforms, Inc.")
    assert "united" not in company_aliases("X", "United States Steel Corporation")


def test_detect_filters(store):
    def f(q):
        return detect_filters(q, store.companies(), store.years)

    assert f("What was Acme's revenue in 2023?") == Filters(["ACME"], 2023)
    assert f("revenue of ACME for fiscal 2022") == Filters(["ACME"], 2022)
    assert f("Compare Acme and Borealis supply chain risks").tickers == ["ACME", "BORE"]
    assert f("What is the supply chain risk?") == Filters([], None)
    assert f("Acme revenue in 2019").year is None  # no such filing, so no year filter
    assert f("what about acme robotics").tickers == ["ACME"]
    assert f("BORE risks").tickers == ["BORE"]
    assert f("bore the risks").tickers == []  # tickers only match in capitals


def test_search_filters_stop_cross_company_mixups(retriever):
    q = "What was the total net revenue in fiscal 2023 for Borealis?"
    plain, _ = retriever.search(q, k=3, rerank=False, use_filters=False)
    filtered, flt = retriever.search(q, k=3, rerank=False, use_filters=True)
    assert flt.tickers == ["BORE"]
    assert all(h.ticker == "BORE" and h.fiscal_year == 2023 for h in filtered)
    assert any("642,700" in h.text for h in filtered)
    assert plain  # without filters it still works, just less precisely


def test_multi_company_question_gets_both(retriever):
    hits, flt = retriever.search("Compare supply chain risk at Acme and Borealis", k=6, rerank=False)
    assert {h.ticker for h in hits} == {"ACME", "BORE"}


def test_all_modes_work(retriever):
    for mode in ("bm25", "dense", "hybrid"):
        hits, _ = retriever.search("lithium battery supply chain", k=3, mode=mode, rerank=False)
        assert hits and "lithium" in hits[0].text.lower()


def test_reranker_changes_order(store):
    class Reverse:
        def score(self, query, texts):
            return [float(i) for i in range(len(texts))]  # prefers the last passage

    from filing_analyst.retrieve import Retriever
    r = Retriever(store, HashingEmbedder(), Reverse())
    base, _ = r.search("net revenue", k=5, rerank=False)
    out, _ = r.search("net revenue", k=5, rerank=True)
    assert out[0].id != base[0].id
    assert [h.score for h in out] == sorted([h.score for h in out], reverse=True)


# ------------------------------------------------------------ answers

class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.seen = None

    def complete(self, system, user):
        self.seen = (system, user)
        return self.reply


def passages():
    return [hit(1, "Total net revenue increased 11 percent to $1,250,400 thousand in fiscal 2023."),
            hit(2, "Net income was $88,100 thousand.")]


def test_numbers_in_ignores_small_numbers_and_citations():
    assert numbers_in("Revenue was $1,250,400 thousand [1], up 11%, in 2023.") == ["1250400", "11", "2023"]
    assert numbers_in("There are 3 segments") == []
    assert numbers_in("margin of 41.5%") == ["41.5"]


def test_good_answer_verifies():
    ans = answer_question("q", passages(), FakeLLM("Revenue was $1,250,400 thousand [1]. Net income was $88,100 thousand [2]."))
    assert ans.verified and ans.cited == [1, 2] and not ans.refused


def test_invented_number_is_flagged():
    ans = answer_question("q", passages(), FakeLLM("Revenue was $1,350,400 thousand [1]."))
    assert not ans.verified
    assert "1350400" in ans.problems[0]


def test_number_from_uncited_passage_is_flagged():
    ans = answer_question("q", passages(), FakeLLM("Net income was $88,100 thousand [1]."))
    assert not ans.verified  # 88,100 is in passage 2, but only passage 1 is cited


def test_bad_or_missing_citations_are_flagged():
    assert "do not exist" in answer_question("q", passages(), FakeLLM("Revenue grew [7].")).problems[0]
    assert any("no citations" in p for p in answer_question("q", passages(), FakeLLM("Revenue grew.")).problems)
    assert not verify("[12]", passages())[1] == []  # "12" is a citation marker, not a number to check


def test_refusals():
    ans = answer_question("q", passages(), FakeLLM(NOT_FOUND))
    assert ans.refused and ans.text == NOT_FOUND
    assert answer_question("q", [], FakeLLM("should not be called")).refused


def test_prompt_numbers_the_passages():
    llm = FakeLLM(NOT_FOUND)
    answer_question("What was revenue?", passages(), llm)
    system, user = llm.seen
    assert "[1] Acme 2023 10-K, Item 7 MD&A" in user and "[2]" in user and user.endswith("Question: What was revenue?")
    assert NOT_FOUND in system


# ------------------------------------------------------------ evaluation

def test_score_ranking_and_metrics():
    q = Question("q", "ACME", 2023, ["1,250,400"])
    good = hit(1, "revenue 1,250,400")
    wrong_company = hit(2, "revenue 1,250,400", ticker="BORE")
    wrong_year = hit(3, "revenue 1,250,400", year=2022)
    no_number = hit(4, "nothing")
    assert is_relevant(good, q)
    assert not any(is_relevant(h, q) for h in (wrong_company, wrong_year, no_number))
    assert score_ranking([no_number, wrong_year, good], q, 5) == (True, pytest.approx(1 / 3))
    assert score_ranking([no_number, wrong_year, good], q, 2) == (False, 0.0)
    res = evaluate([q, q], lambda text: [good] if text == "q" else [], 5)
    assert res["recall@5"] == 1.0 and res["mrr@5"] == 1.0
    res = evaluate([q, Question("other", "ACME", 2023, ["zzz"])], lambda text: [good], 5)
    assert res["recall@5"] == 0.5 and res["misses"] == ["other"]


def test_load_questions(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('# comment\n{"question":"a","ticker":"acme","year":"2023","gold":["1"]}\n\n')
    qs = load_questions(str(p))
    assert qs == [Question("a", "ACME", 2023, ["1"])]
    p.write_text('{"question":"a","ticker":"x","year":1,"gold":[]}\n')
    with pytest.raises(ValueError):
        load_questions(str(p))


def test_retrieval_eval_end_to_end(store, retriever, tmp_path):
    from filing_analyst.evaluate import run_eval

    p = tmp_path / "q.jsonl"
    rows = [
        {"question": "What was Acme total net revenue in fiscal 2023?", "ticker": "ACME", "year": 2023, "gold": ["1,250,400"]},
        {"question": "What was Borealis net income in fiscal 2023?", "ticker": "BORE", "year": 2023, "gold": ["30,500"]},
        {"question": "What was Acme total net revenue in fiscal 2022?", "ticker": "ACME", "year": 2022, "gold": ["1,125,360"]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    results = run_eval(str(p), store, HashingEmbedder(), k=3, with_rerank=False, log=lambda m: None)
    by_name = {r["setup"]: r for r in results}
    assert len(results) == 4
    # With the company and year filter, every question here is answerable from the top 3.
    assert by_name["hybrid + company/year filter"]["recall@3"] == 1.0
    assert by_name["hybrid + company/year filter"]["recall@3"] >= by_name["hybrid (keyword + vector)"]["recall@3"]
