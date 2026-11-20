"""Uses the real embedding and reranking models (a one time download of about
200 MB). Skipped automatically if they cannot be loaded."""
import pytest

from filing_analyst.ingest import ingest_folder
from filing_analyst.retrieve import Retriever
from filing_analyst.store import Store


@pytest.fixture(scope="module")
def real(filings_dir, tmp_path_factory):
    try:
        from filing_analyst.embed import FastEmbedder
        from filing_analyst.retrieve import CrossEncoderReranker

        emb, rer = FastEmbedder(), CrossEncoderReranker()
    except Exception as e:  # no network, no model files
        pytest.skip(f"models unavailable: {e}")
    store = Store(str(tmp_path_factory.mktemp("real_index")))
    ingest_folder(filings_dir, store, emb, log=lambda m: None)
    return Retriever(store, emb, rer)


def test_paraphrase_found_by_meaning(real):
    # No word in common with the passage except "battery": keyword search alone is weak here.
    q = "What could go wrong for Acme if its power cell vendors cannot deliver on time?"
    hits, _ = real.search(q, k=3, mode="dense", rerank=False)
    assert any("lithium" in h.text.lower() and h.ticker == "ACME" for h in hits)


def test_reranker_puts_the_answer_first(real):
    hits, flt = real.search("What was Acme's total net revenue in fiscal 2023?", k=3)
    assert flt.tickers == ["ACME"] and flt.year == 2023
    assert "1,250,400" in hits[0].text


def test_filter_beats_wrong_company(real):
    hits, _ = real.search("Borealis supply chain disruption risk", k=3)
    assert hits[0].ticker == "BORE"
