"""Read the tagged numbers (inline XBRL) inside a 10-K, to build eval questions.

Since 2019 every 10-K is filed as "inline XBRL": the HTML people read, with each
official number wrapped in a tag that says what it is and which period it
covers, for example

    <ix:nonFraction name="us-gaap:Revenues" contextRef="c-1">416,161</ix:nonFraction>

where context c-1 says "the year ended 2025-09-27, whole company". These tags
are the SEC's structured data, so questions built from them have objective
answers, and they come straight from the files already downloaded: no network.

The answer kept for each figure is the text exactly as the filing prints it
("416,161"), which is what a retrieved passage must contain."""
from __future__ import annotations

import re
import warnings
import zlib
from dataclasses import dataclass
from datetime import date

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning


@dataclass(frozen=True)
class Metric:
    name: str
    tags: tuple[str, ...]  # tried in order; companies use different tags for the same line
    kind: str  # "period" (over the year, like revenue) or "instant" (at year end, like assets)
    wordings: tuple[str, ...]  # several phrasings, so the eval does not reward one wording


METRICS = [
    Metric("revenue", ("us-gaap:Revenues", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                       "us-gaap:SalesRevenueNet"), "period",
           ("What was {company}'s total revenue for fiscal {year}?",
            "How much did {company} bring in from sales in fiscal {year}?",
            "What were {company}'s total net sales in its {year} fiscal year?")),
    Metric("cost of revenue", ("us-gaap:CostOfRevenue", "us-gaap:CostOfGoodsAndServicesSold"), "period",
           ("What was {company}'s total cost of revenue in fiscal {year}?",
            "How much did it cost {company} to make and deliver what it sold in fiscal {year}?",
            "What were {company}'s total cost of sales for fiscal {year}?")),
    Metric("gross profit", ("us-gaap:GrossProfit",), "period",
           ("What was {company}'s gross profit for fiscal {year}?",
            "How much gross margin, in dollars, did {company} earn in fiscal {year}?",
            "What was {company}'s revenue minus cost of revenue in fiscal {year}?")),
    Metric("research and development", ("us-gaap:ResearchAndDevelopmentExpense",), "period",
           ("How much did {company} spend on research and development in fiscal {year}?",
            "What was {company}'s R&D expense for fiscal {year}?",
            "What were {company}'s research and development costs in its {year} fiscal year?")),
    Metric("operating income", ("us-gaap:OperatingIncomeLoss",), "period",
           ("What was {company}'s operating income for fiscal {year}?",
            "How much profit did {company} make from operations in fiscal {year}?",
            "What was {company}'s income from operations in its {year} fiscal year?")),
    Metric("income tax", ("us-gaap:IncomeTaxExpenseBenefit",), "period",
           ("What was {company}'s provision for income taxes in fiscal {year}?",
            "How much income tax expense did {company} record for fiscal {year}?",
            "What did {company} pay in income taxes, as an expense, in its {year} fiscal year?")),
    Metric("net income", ("us-gaap:NetIncomeLoss",), "period",
           ("What was {company}'s net income for fiscal {year}?",
            "How much did {company} earn after all expenses and taxes in fiscal {year}?",
            "What was {company}'s bottom line profit in its {year} fiscal year?")),
    Metric("operating cash flow", ("us-gaap:NetCashProvidedByUsedInOperatingActivities",), "period",
           ("How much cash did {company}'s operations generate in fiscal {year}?",
            "What was {company}'s net cash from operating activities for fiscal {year}?",
            "What was {company}'s operating cash flow in its {year} fiscal year?")),
    Metric("capital spending", ("us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
                                "us-gaap:PaymentsToAcquireProductiveAssets"), "period",
           ("How much did {company} spend on property and equipment in fiscal {year}?",
            "What were {company}'s capital expenditures for fiscal {year}?",
            "How much did {company} invest in buildings, equipment and data centers in fiscal {year}?")),
    Metric("cash", ("us-gaap:CashAndCashEquivalentsAtCarryingValue",), "instant",
           ("How much cash and cash equivalents did {company} hold at the end of fiscal {year}?",
            "What was {company}'s cash balance at the close of fiscal {year}?",
            "How much cash did {company} have on hand when its {year} fiscal year ended?")),
    Metric("total assets", ("us-gaap:Assets",), "instant",
           ("What were {company}'s total assets at the end of fiscal {year}?",
            "What was everything {company} owned worth on its balance sheet at the end of fiscal {year}?",
            "What was {company}'s total asset value when its {year} fiscal year ended?")),
    Metric("total liabilities", ("us-gaap:Liabilities",), "instant",
           ("What were {company}'s total liabilities at the end of fiscal {year}?",
            "How much did {company} owe in total at the close of fiscal {year}?",
            "What was the sum of {company}'s liabilities when its {year} fiscal year ended?")),
    Metric("shareholders' equity", ("us-gaap:StockholdersEquity",), "instant",
           ("What was {company}'s total shareholders' equity at the end of fiscal {year}?",
            "What was {company}'s book value (stockholders' equity) at the close of fiscal {year}?",
            "How much equity did {company}'s shareholders have when its {year} fiscal year ended?")),
]

MIN_DIGITS = 5  # shorter numbers (like 1,234) appear all over a filing, so finding one proves nothing
NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+|\d+")


@dataclass
class _Context:
    start: date | None
    end: date
    whole_company: bool  # no segment: not just one product, region or business line


def _date(text: str | None) -> date | None:
    try:
        return date.fromisoformat((text or "").strip())
    except ValueError:
        return None


def _contexts(soup: BeautifulSoup) -> dict[str, _Context]:
    out = {}
    for c in soup.find_all("xbrli:context"):
        period = c.find("xbrli:period")
        if period is None or not c.get("id"):
            continue
        instant = period.find("xbrli:instant")
        start_tag, end_tag = period.find("xbrli:startdate"), period.find("xbrli:enddate")
        if instant is not None:
            start, end = None, _date(instant.get_text())
        elif start_tag is not None and end_tag is not None:
            start, end = _date(start_tag.get_text()), _date(end_tag.get_text())
            if start is None:
                continue
        else:
            continue
        if end is None:
            continue
        out[c["id"]] = _Context(start, end, whole_company=c.find("xbrli:segment") is None)
    return out


def _matches(ctx: _Context | None, kind: str, year_end: date) -> bool:
    if ctx is None or not ctx.whole_company or ctx.end != year_end:
        return False
    if kind == "instant":
        return ctx.start is None
    return ctx.start is not None and 340 <= (ctx.end - ctx.start).days <= 380  # a full year, not a quarter


def _shown(tag) -> str | None:
    """The number as printed, if it is long enough to be a safe answer."""
    text = " ".join(tag.get_text(" ").split())
    m = NUMBER.fullmatch(text)
    if not m or len(text.replace(",", "")) < MIN_DIGITS:
        return None
    return text


def read_facts(html: str) -> dict[str, list[str]]:
    """{metric name: [the number as printed]} for the year this filing covers,
    for the whole company. Figures that cannot be read safely are left out."""
    try:
        with warnings.catch_warnings():
            # Filings start with an XML declaration; the HTML parser handles them fine.
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(html, "lxml")
    except Exception:  # noqa: BLE001  (a file that cannot be parsed has no usable facts)
        return {}
    contexts = _contexts(soup)
    period = soup.find("ix:nonnumeric", attrs={"name": "dei:DocumentPeriodEndDate"})
    year_ctx = contexts.get(period.get("contextref", "")) if period is not None else None
    if year_ctx is None:
        return {}
    year_end = year_ctx.end  # from the tag's context, whatever words the page shows
    by_name: dict[str, list] = {}
    for t in soup.find_all("ix:nonfraction"):
        by_name.setdefault(t.get("name", ""), []).append(t)
    facts = {}
    for m in METRICS:
        for tag in m.tags:
            shown = [s for t in by_name.get(tag, []) if _matches(contexts.get(t.get("contextref", "")), m.kind, year_end)
                     and (s := _shown(t))]
            if shown:
                facts[m.name] = list(dict.fromkeys(shown))
                break
    return facts


def build_questions(facts: dict[str, list[str]], ticker: str, company: str, year: int) -> list[dict]:
    """One question per figure, in the eval's format. The wording is picked from
    the company, year and figure, so it varies across questions but a rebuilt
    file is always the same."""
    rows = []
    for m in METRICS:
        if m.name not in facts:
            continue
        pick = zlib.crc32(f"{ticker}|{year}|{m.name}".encode()) % len(m.wordings)
        rows.append({"question": m.wordings[pick].format(company=company, year=year), "ticker": ticker,
                     "year": year, "gold": facts[m.name], "group": m.name})
    return rows
