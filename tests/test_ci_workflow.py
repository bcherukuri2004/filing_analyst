"""Checks on the CI workflow and the Docker smoke test, without running them.

The smoke test was added next to the retrieval quality gate. These tests make
sure it stays wired up, and that the gate is never quietly weakened: its job,
its command and its minimum scores are pinned here (the scores may be raised,
never lowered)."""
import json
import os
import stat
import subprocess
import sys

import yaml

from filing_analyst.embed import HashingEmbedder
from filing_analyst.ingest import ingest_folder
from filing_analyst.store import Store

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.path.join(ROOT, "scripts", "docker_smoke.sh")

# The gate as it was when the smoke test was added.
GATE_FLOOR = {"min_questions": 110,
              "setups": {"vector search only": {"recall@5": 0.78, "mrr@5": 0.55},
                         "hybrid + filter + reranker": {"recall@5": 0.50, "mrr@5": 0.27}}}


def workflow() -> dict:
    with open(os.path.join(ROOT, ".github", "workflows", "ci.yml")) as f:
        return yaml.safe_load(f)


def run_lines(job: dict) -> list[str]:
    return [s["run"] for s in job["steps"] if "run" in s]


def test_the_three_jobs_exist_and_only_read_the_repository():
    wf = workflow()
    assert {"tests", "docker-smoke", "eval-gate"} <= set(wf["jobs"])
    assert wf["permissions"] == {"contents": "read"}


def test_the_smoke_job_runs_the_smoke_script_after_the_tests():
    job = workflow()["jobs"]["docker-smoke"]
    assert job["needs"] == "tests"
    assert run_lines(job) == ["scripts/docker_smoke.sh"]
    assert "SEC_USER_AGENT" not in json.dumps(job)  # it uses fictional filings, no secrets


def test_the_eval_gate_still_runs_on_the_wide_eval_with_the_gate_file():
    job = workflow()["jobs"]["eval-gate"]
    assert job["needs"] == "tests"
    assert "python -m filing_analyst eval eval/questions_wide.jsonl --gate eval/gate.json" in run_lines(job)
    assert "continue-on-error" not in json.dumps(job)  # a failed gate must fail the run


def test_the_gate_minimums_were_not_lowered():
    with open(os.path.join(ROOT, "eval", "gate.json")) as f:
        gate = json.load(f)
    assert gate["min_questions"] >= GATE_FLOOR["min_questions"]
    for setup, minimums in GATE_FLOOR["setups"].items():
        for metric, floor in minimums.items():
            assert gate["setups"][setup][metric] >= floor, (setup, metric)


def test_the_smoke_script_is_safe_to_run_anywhere():
    text = open(SMOKE).read()
    assert os.stat(SMOKE).st_mode & stat.S_IXUSR
    assert "set -euo pipefail" in text
    assert "trap cleanup EXIT" in text and "down --volumes" in text  # always cleans up
    assert "unset ANTHROPIC_API_KEY" in text and '"/ask"' not in text  # never spends credits
    assert "openssl rand" in text  # throwaway passwords, nothing fixed


def test_the_sample_filings_index_and_hold_the_answer_the_smoke_test_checks(tmp_path):
    out = tmp_path / "filings"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "write_sample_filings.py"), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert sorted(os.listdir(out)) == ["ACME_2022.htm", "ACME_2023.htm", "BORE_2023.htm", "companies.json"]
    store = Store(str(tmp_path / "index"))
    assert ingest_folder(str(out), store, HashingEmbedder(), log=lambda m: None) > 0
    assert store.any_contains("ACME", 2023, ["1,250,400"])


def test_the_sample_filings_script_needs_a_folder():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "write_sample_filings.py")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "usage" in r.stderr
