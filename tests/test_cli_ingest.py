"""Tests for `ingest --if-empty` and for the index address coming from the
FILING_ANALYST_INDEX setting.

In Docker the index is built by a one time step that runs on every
`docker compose up`. It must do the slow work only the first time and skip it
afterwards, and it must never skip when the existing index was made with a
different embedding model (its vectors would not match the questions)."""
import os
import subprocess
import sys

import pytest

from filing_analyst import cli
from filing_analyst.embed import DEFAULT_MODEL, HashingEmbedder
from filing_analyst.store import open_store


class CountingEmbedder(HashingEmbedder):
    """Fast stand in that claims to be the real model, and counts its use."""

    calls = 0

    def __init__(self, name=DEFAULT_MODEL):
        super().__init__()
        self.name = name

    def embed_passages(self, texts):
        CountingEmbedder.calls += 1
        return super().embed_passages(texts)


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    CountingEmbedder.calls = 0
    monkeypatch.setattr(cli, "_embedder", lambda: CountingEmbedder())
    monkeypatch.delenv("FILING_ANALYST_INDEX", raising=False)


def ingest(*args):
    cli.main(list(args))


def test_if_empty_builds_an_empty_index(filings_dir, tmp_path, capsys):
    ingest("--index", str(tmp_path), "ingest", str(filings_dir), "--if-empty")
    assert open_store(str(tmp_path)).count() > 0
    assert CountingEmbedder.calls == 1
    assert "Indexed" in capsys.readouterr().out


def test_if_empty_skips_an_index_that_is_already_built(filings_dir, tmp_path, capsys, monkeypatch):
    ingest("--index", str(tmp_path), "ingest", str(filings_dir))
    before = open_store(str(tmp_path)).count()

    def no_model():
        raise AssertionError("the model must not even be loaded when there is nothing to do")
    monkeypatch.setattr(cli, "_embedder", no_model)
    capsys.readouterr()
    ingest("--index", str(tmp_path), "ingest", str(filings_dir), "--if-empty")
    assert "already has" in capsys.readouterr().out
    assert open_store(str(tmp_path)).count() == before


def test_without_the_option_it_always_rebuilds(filings_dir, tmp_path):
    ingest("--index", str(tmp_path), "ingest", str(filings_dir))
    ingest("--index", str(tmp_path), "ingest", str(filings_dir))
    assert CountingEmbedder.calls == 2


def test_an_index_from_another_model_is_rebuilt(filings_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_embedder", lambda: CountingEmbedder(name="some-older-model"))
    ingest("--index", str(tmp_path), "ingest", str(filings_dir))
    monkeypatch.setattr(cli, "_embedder", lambda: CountingEmbedder())
    ingest("--index", str(tmp_path), "ingest", str(filings_dir), "--if-empty")
    assert CountingEmbedder.calls == 2
    assert open_store(str(tmp_path)).embedder_name() == DEFAULT_MODEL


def test_if_empty_with_no_filings_still_fails_clearly(tmp_path):
    empty = tmp_path / "no_filings"
    empty.mkdir()
    with pytest.raises(SystemExit) as e:
        ingest("--index", str(tmp_path / "index"), "ingest", str(empty), "--if-empty")
    assert "No files" in str(e.value)


def test_the_index_address_can_come_from_the_setting(filings_dir, tmp_path, monkeypatch):
    # So a password inside a postgresql:// address never has to be typed on a
    # command line, where other programs on the machine can read it.
    monkeypatch.setenv("FILING_ANALYST_INDEX", str(tmp_path / "from_env"))
    ingest("ingest", str(filings_dir))
    assert open_store(str(tmp_path / "from_env")).count() > 0


def test_an_explicit_index_beats_the_setting(filings_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FILING_ANALYST_INDEX", str(tmp_path / "from_env"))
    ingest("--index", str(tmp_path / "explicit"), "ingest", str(filings_dir))
    assert open_store(str(tmp_path / "explicit")).count() > 0
    assert not os.path.exists(tmp_path / "from_env" / "filings.db")


def test_without_the_setting_the_default_folder_is_used(monkeypatch):
    monkeypatch.delenv("FILING_ANALYST_INDEX", raising=False)
    assert cli.default_index() == "index"


def test_an_unreachable_database_fails_without_showing_the_password(tmp_path):
    env = {**os.environ, "FILING_ANALYST_INDEX":
           "postgresql://filing:Sup3rSecretPw@127.0.0.1:1/filing_test?connect_timeout=2"}
    r = subprocess.run([sys.executable, "-m", "filing_analyst", "ingest", str(tmp_path), "--if-empty"],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "Could not connect to Postgres" in r.stderr
    assert "Sup3rSecretPw" not in r.stdout + r.stderr
    assert "Traceback" not in r.stderr  # one plain line, not a Python error report
