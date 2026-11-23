"""Tests for reading the tagged numbers (inline XBRL) inside a 10-K, and for the
questions built from them.

A 10-K repeats many numbers: last year's column, figures for one segment, the
fourth quarter alone. Only the company-wide figure for the year the filing
covers is the right answer, so most tests here plant a wrong number next to the
right one and check that the wrong one is never chosen."""
import json

import pytest

from filing_analyst.xbrl import METRICS, build_questions, read_facts


def context(cid: str, end: str, start: str | None = None, segment: bool = False) -> str:
    period = (f"<xbrli:startDate>{start}</xbrli:startDate><xbrli:endDate>{end}</xbrli:endDate>" if start
              else f"<xbrli:instant>{end}</xbrli:instant>")
    seg = ('<xbrli:segment><xbrldi:explicitMember dimension="srt:ProductOrServiceAxis">'
           "us-gaap:ProductMember</xbrldi:explicitMember></xbrli:segment>") if segment else ""
    return (f'<xbrli:context id="{cid}"><xbrli:entity><xbrli:identifier scheme="http://www.sec.gov/CIK">1'
            f"</xbrli:identifier>{seg}</xbrli:entity><xbrli:period>{period}</xbrli:period></xbrli:context>")


def fact(name: str, cid: str, shown: str, **attrs) -> str:
    extra = "".join(f' {k}="{v}"' for k, v in attrs.items())
    return f'<ix:nonFraction name="{name}" contextRef="{cid}" unitRef="usd"{extra}>{shown}</ix:nonFraction>'


def filing(*facts: str, end_date: str = "2025-09-27", extra_contexts: str = "") -> str:
    contexts = (context("year", "2025-09-27", "2024-09-29") + context("prior", "2024-09-28", "2023-10-01")
                + context("q4", "2025-09-27", "2025-06-29") + context("seg", "2025-09-27", "2024-09-29", True)
                + context("now", "2025-09-27") + context("then", "2024-09-28") + extra_contexts)
    period = (f'<ix:nonNumeric name="dei:DocumentPeriodEndDate" contextRef="year">{end_date}</ix:nonNumeric>'
              if end_date else "")
    return (f"<html><body><div style='display:none'><ix:header><ix:resources>{contexts}</ix:resources>"
            f"</ix:header></div><p>For the fiscal year ended {period}</p><table><tr><td>"
            + "</td><td>".join(facts) + "</td></tr></table></body></html>")


REV = "us-gaap:Revenues"


def test_reads_the_company_wide_figure_for_the_filing_year():
    html = filing(fact(REV, "year", "416,161"), fact(REV, "prior", "391,035"),
                  fact(REV, "seg", "307,003"), fact(REV, "q4", "102,466"))
    assert read_facts(html)["revenue"] == ["416,161"]


def test_balance_sheet_figures_use_the_year_end_date():
    html = filing(fact("us-gaap:Assets", "now", "359,241"), fact("us-gaap:Assets", "then", "364,980"))
    assert read_facts(html)["total assets"] == ["359,241"]


def test_a_duration_figure_is_not_read_from_a_single_date_and_back():
    # Revenue tagged on a balance-sheet date, or assets over a period, are mistakes to skip.
    html = filing(fact(REV, "now", "416,161"), fact("us-gaap:Assets", "year", "359,241"))
    facts = read_facts(html)
    assert "revenue" not in facts and "total assets" not in facts


def test_the_first_tag_that_has_a_value_wins():
    # Companies use different tags for the same line; the first listed one is preferred.
    html = filing(fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "year", "111,111"),
                  fact(REV, "year", "222,222"))
    assert read_facts(html)["revenue"] == ["222,222"]


def test_every_way_the_filing_prints_the_number_is_kept():
    html = filing(fact(REV, "year", "416,161"), fact(REV, "year", "416,161"), fact(REV, "year", "416.2", scale="9"))
    assert read_facts(html)["revenue"] == ["416,161"]  # "416.2" is too short to be a safe answer


@pytest.mark.parametrize("shown", ["1,234", "—", "", "12", "(5)"])
def test_numbers_too_short_or_empty_to_check_are_skipped(shown):
    # A short number like 1,234 appears all over a filing, so finding it proves nothing.
    assert "revenue" not in read_facts(filing(fact(REV, "year", shown)))


def test_negative_numbers_keep_their_digits():
    html = filing(fact("us-gaap:IncomeTaxExpenseBenefit", "year", "20,719", sign="-"))
    assert read_facts(html)["income tax"] == ["20,719"]


def test_nested_tags_and_spacing_are_handled():
    html = filing(f'<ix:nonFraction name="{REV}" contextRef="year"> <span>416,161</span>\n</ix:nonFraction>')
    assert read_facts(html)["revenue"] == ["416,161"]


def test_unknown_contexts_and_concepts_are_ignored():
    html = filing(fact(REV, "no-such-context", "416,161"), fact("us-gaap:SomethingElse", "year", "999,999"))
    assert read_facts(html) == {}


@pytest.mark.parametrize("html", [
    "",                                                     # empty file
    "<html><body>no tags at all</body></html>",             # an old filing without inline XBRL
    filing(fact(REV, "year", "416,161"), end_date=""),      # no period end date
    "<html><ix:nonFraction name='us-gaap:Revenues'>416,161",  # broken, unclosed markup
])
def test_files_without_usable_tags_give_nothing_and_do_not_crash(html):
    assert read_facts(html) == {}


def test_the_period_end_comes_from_the_tag_not_the_words_shown():
    # Real filings show "September 27, 2025"; the exact date is in the tag's context.
    for shown in ("September&#160;27, 2025", "Sept. 27", "anything at all"):
        assert read_facts(filing(fact(REV, "year", "416,161"), end_date=shown))["revenue"] == ["416,161"]


# ------------------------------------------------------------ questions

def test_questions_have_the_eval_format_and_a_type():
    rows = build_questions({"revenue": ["416,161"], "total assets": ["359,241"]}, "AAPL", "Apple", 2025)
    assert {r["group"] for r in rows} == {"revenue", "total assets"}
    for r in rows:
        assert set(r) == {"question", "ticker", "year", "gold", "group"}
        assert r["ticker"] == "AAPL" and r["year"] == 2025 and "Apple" in r["question"] and "2025" in r["question"]
        json.dumps(r)


def test_wording_varies_but_is_the_same_every_run():
    # Several wordings per figure, so the eval does not reward one phrasing;
    # chosen from the company, year and figure, so the file never changes by chance.
    facts = {m.name: ["416,161"] for m in METRICS}
    a = build_questions(facts, "AAPL", "Apple", 2025)
    assert a == build_questions(facts, "AAPL", "Apple", 2025)
    # Same company name and year every time, so only the wording can differ.
    wordings = {q["question"] for t in ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA")
                for q in build_questions(facts, t, "Co", 2025) if q["group"] == "revenue"}
    assert len(wordings) > 1


def test_every_figure_has_several_wordings_that_name_company_and_year():
    for m in METRICS:
        assert len(m.wordings) >= 3, m.name
        for w in m.wordings:
            assert "{company}" in w and "{year}" in w, w
