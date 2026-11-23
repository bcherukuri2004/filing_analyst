"""Tests for the eval gate: the check that fails CI when search quality drops.

A gate is only useful if it fails when it should, so most of these try to get a
bad result past it: a score just under the line, too few questions graded, a
typo in a setup name, a gate that checks nothing."""
import json
import os
import subprocess
import sys

import pytest

from filing_analyst.evaluate import check_gate, load_gate

ROWS = [
    {"setup": "keyword search only (BM25)", "n": 40, "recall@5": 0.0, "mrr@5": 0.0},
    {"setup": "vector search only", "n": 40, "recall@5": 1.0, "mrr@5": 0.835},
    {"setup": "hybrid + filter + reranker", "n": 40, "recall@5": 0.8, "mrr@5": 0.415},
]
GATE = {"min_questions": 35,
        "setups": {"vector search only": {"recall@5": 0.95, "mrr@5": 0.75},
                   "hybrid + filter + reranker": {"recall@5": 0.70}}}


def rows_with(setup: str, **changes) -> list[dict]:
    return [{**r, **changes} if r["setup"] == setup else r for r in ROWS]


def test_good_results_pass():
    assert check_gate(ROWS, GATE) == []


def test_exactly_on_the_line_passes():
    assert check_gate(rows_with("vector search only", **{"recall@5": 0.95}), GATE) == []


def test_just_under_the_line_fails_and_says_by_how_much():
    problems = check_gate(rows_with("vector search only", **{"mrr@5": 0.749}), GATE)
    assert len(problems) == 1
    assert "vector search only" in problems[0] and "0.749" in problems[0] and "0.75" in problems[0]


def test_every_failure_is_reported_not_just_the_first():
    rows = rows_with("vector search only", **{"recall@5": 0.5, "mrr@5": 0.1})
    rows = [{**r, "recall@5": 0.1} if r["setup"] == "hybrid + filter + reranker" else r for r in rows]
    assert len(check_gate(rows, GATE)) == 3


def test_too_few_graded_questions_fails():
    # If the parser breaks, questions whose answer text cannot be found are left
    # out, and the rest could still score well. The gate must notice.
    rows = [{**r, "n": 20} for r in ROWS]
    problems = check_gate(rows, GATE)
    assert any("20" in p and "35" in p for p in problems)


def test_a_setup_the_eval_did_not_run_fails():
    # A typo in the gate file, or running without the reranker, must not pass silently.
    gate = {"min_questions": 1, "setups": {"vector serach only": {"recall@5": 0.5}}}
    assert any("vector serach only" in p for p in check_gate(ROWS, gate))
    no_rerank = [r for r in ROWS if r["setup"] != "hybrid + filter + reranker"]
    assert any("hybrid + filter + reranker" in p for p in check_gate(no_rerank, GATE))


def test_a_metric_the_eval_did_not_report_fails():
    gate = {"min_questions": 1, "setups": {"vector search only": {"recall@10": 0.5}}}
    assert any("recall@10" in p for p in check_gate(ROWS, gate))


@pytest.mark.parametrize("gate, message", [
    ({"min_questions": 1, "setups": {}}, "checks nothing"),
    ({"min_questions": 1}, "checks nothing"),
    ({"min_questions": 1, "setups": {"vector search only": {}}}, "checks nothing"),
    ({"min_questions": 1, "setups": {"vector search only": {"recall@5": 1.5}}}, "between 0 and 1"),
    ({"min_questions": 1, "setups": {"vector search only": {"recall@5": -0.1}}}, "between 0 and 1"),
    ({"min_questions": 1, "setups": {"vector search only": {"recall@5": "high"}}}, "between 0 and 1"),
    ({"min_questions": 1, "setups": {"vector search only": {"recall@5": True}}}, "between 0 and 1"),
    ({"min_questions": 0, "setups": {"vector search only": {"recall@5": 0.5}}}, "min_questions"),
    ({"setups": {"vector search only": {"recall@5": 0.5}}}, "min_questions"),
    ({"min_questions": 2.5, "setups": {"vector search only": {"recall@5": 0.5}}}, "min_questions"),
])
def test_a_broken_gate_file_is_refused(tmp_path, gate, message):
    p = tmp_path / "gate.json"
    p.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match=message):
        load_gate(str(p))


@pytest.mark.parametrize("text", ["", "{not json", "[1, 2]"])
def test_a_gate_file_that_is_not_a_json_object_is_refused(tmp_path, text):
    p = tmp_path / "gate.json"
    p.write_text(text)
    with pytest.raises(ValueError):
        load_gate(str(p))


def test_the_committed_gate_file_is_valid():
    gate = load_gate("eval/gate.json")
    assert gate["min_questions"] >= 30


def test_eval_command_exits_with_an_error_when_the_gate_fails(tmp_path, filings_dir):
    # End to end through the real command line and real models, on the small
    # fictional filings. CI relies on the exit code, so that is what is checked.
    questions = tmp_path / "q.jsonl"
    questions.write_text(json.dumps({"question": "What was Acme's total net revenue in 2023?",
                                     "ticker": "ACME", "year": 2023, "gold": ["1,250,400"]}) + "\n")
    run = [sys.executable, "-m", "filing_analyst", "--index", str(tmp_path / "index")]
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    subprocess.run([*run, "ingest", filings_dir], check=True, capture_output=True, env=env)

    def eval_with(gate: dict) -> subprocess.CompletedProcess:
        g = tmp_path / "gate.json"
        g.write_text(json.dumps(gate))
        return subprocess.run([*run, "eval", str(questions), "--no-rerank", "--gate", str(g)],
                              capture_output=True, text=True, env=env)

    ok = eval_with({"min_questions": 1, "setups": {"vector search only": {"recall@5": 0.5}}})
    assert ok.returncode == 0 and "Gate passed" in ok.stdout, ok.stdout + ok.stderr
    failed = eval_with({"min_questions": 5, "setups": {"vector search only": {"recall@5": 0.5}}})
    assert failed.returncode == 1, failed.stdout + failed.stderr
    assert "Gate FAILED" in failed.stdout and "only 1 question" in failed.stdout
