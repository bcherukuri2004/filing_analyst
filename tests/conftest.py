import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures import ACME_2022, ACME_2023, BOREALIS_2023  # noqa: E402

from filing_analyst.embed import HashingEmbedder  # noqa: E402
from filing_analyst.ingest import ingest_folder  # noqa: E402
from filing_analyst.retrieve import Retriever  # noqa: E402
from filing_analyst.store import Store  # noqa: E402


@pytest.fixture(scope="session")
def filings_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("filings")
    (d / "ACME_2023.htm").write_text(ACME_2023)
    (d / "ACME_2022.htm").write_text(ACME_2022)
    (d / "BORE_2023.htm").write_text(BOREALIS_2023)
    (d / "companies.json").write_text(json.dumps({"ACME": "Acme Robotics, Inc.", "BORE": "Borealis Foods Corporation"}))
    return str(d)


@pytest.fixture(scope="session")
def store(filings_dir, tmp_path_factory):
    s = Store(str(tmp_path_factory.mktemp("index")))
    ingest_folder(filings_dir, s, HashingEmbedder(), log=lambda m: None)
    return s


@pytest.fixture(scope="session")
def retriever(store):
    return Retriever(store, HashingEmbedder())
