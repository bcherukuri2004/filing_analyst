"""Tests for the wider eval: the question file builder, the committed question
file, and the breakdown by kind of question."""
import json
import os
import subprocess
import sys

from test_xbrl import fact, filing

from filing_analyst.evaluate import Question, evaluate, load_questions
from filing_analyst.store import Hit
from filing_analyst.xbrl import METRICS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "make_wide_eval.py")


def run_script(folder, out) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, SCRIPT, str(folder), str(out)], capture_output=True, text=True)


def test_builder_writes_questions_and_skips_files_without_tags(tmp_path):
    d = tmp_path / "filings"
    d.mkdir()
    (d / "ACME_2025.htm").write_text(filing(fact("us-gaap:Revenues", "year", "416,161"),
                                            fact("us-gaap:Assets", "now", "359,241")))
    (d / "OLDCO_2025.htm").write_text("<html><body>Item 7. Old style filing, no tags</body></html>")
    (d / "companies.json").write_text(json.dumps({"ACME": "Acme Robotics, Inc."}))
    r = run_script(d, tmp_path / "q.jsonl")
    assert r.returncode == 0, r.stderr
    assert "OLDCO_2025.htm: no tagged figures found" in r.stdout
    rows = [json.loads(line) for line in (tmp_path / "q.jsonl").read_text().splitlines()]
    assert [(q["group"], q["gold"]) for q in rows] == [("revenue", ["416,161"]), ("total assets", ["359,241"])]
    assert all("Acme Robotics, Inc." in q["question"] for q in rows)
    first = (tmp_path / "q.jsonl").read_text()
    run_script(d, tmp_path / "q.jsonl")
    assert (tmp_path / "q.jsonl").read_text() == first  # rebuilding never changes the file


def test_builder_fails_clearly_when_nothing_can_be_built(tmp_path):
    d = tmp_path / "filings"
    d.mkdir()
    (d / "OLDCO_2025.htm").write_text("<html><body>no tags</body></html>")
    r = run_script(d, tmp_path / "q.jsonl")
    assert r.returncode != 0 and "No questions could be built" in r.stderr


def test_the_committed_wide_question_file_is_sound():
    path = os.path.join(ROOT, "eval", "questions_wide.jsonl")
    questions = load_questions(path)
    assert len(questions) >= 100
    names = {m.name for m in METRICS}
    seen = set()
    for q in questions:
        assert q.group in names
        assert (q.ticker, q.year, q.group) not in seen  # one question per figure per filing
        seen.add((q.ticker, q.year, q.group))
        assert str(q.year) in q.question
        assert all(len(g.replace(",", "")) >= 5 for g in q.gold)
    assert len({q.group for q in questions}) == len(METRICS)  # every kind of figure is asked somewhere
    assert len({q.ticker for q in questions}) == 5


def hit(ticker: str, year: int, text: str) -> Hit:
    return Hit(1, ticker, ticker, year, "7", "MD&A", "text", text)


def test_breakdown_counts_each_kind_of_question():
    qs = [Question("a", "X", 2025, ["11111"], "revenue"), Question("b", "X", 2025, ["22222"], "revenue"),
          Question("c", "X", 2025, ["33333"], "cash")]
    answers = {"a": [hit("X", 2025, "11111")], "b": [hit("X", 2025, "nope")], "c": [hit("X", 2025, "33333")]}
    res = evaluate(qs, lambda text: answers[text], k=5)
    assert res["by_group"] == {"revenue": (1, 2), "cash": (1, 1)}
    assert res["recall@5"] == 2 / 3


def test_old_question_files_without_a_kind_still_load(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps({"question": "q", "ticker": "x", "year": 2025, "gold": ["12345"]}) + "\n")
    assert load_questions(str(p))[0].group == ""
