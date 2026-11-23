"""The same storage tests, run against both stores: SQLite (the default) and
Postgres + pgvector. If the two ever behave differently, a test here fails.

The Postgres run needs a database to test in, given as an address:

    FILING_ANALYST_TEST_PG=postgresql://filing:<password>@127.0.0.1:5432/filing_test pytest

Without it the Postgres run is skipped. The database name must contain "test",
because these tests wipe it."""
import os
from urllib.parse import urlsplit

import pytest

from filing_analyst.chunk import Chunk
from filing_analyst.embed import HashingEmbedder
from filing_analyst.ingest import ingest_folder
from filing_analyst.retrieve import Retriever
from filing_analyst.store import Store, describe_location, open_store

PG = os.environ.get("FILING_ANALYST_TEST_PG", "")


def pg_url() -> str:
    if not PG:
        pytest.skip("set FILING_ANALYST_TEST_PG to run the Postgres tests")
    if "test" not in urlsplit(PG).path:
        pytest.fail("FILING_ANALYST_TEST_PG must name a database with 'test' in it: these tests wipe it")
    return PG


def make_store(kind: str, tmp_path_factory):
    if kind == "sqlite":
        return open_store(str(tmp_path_factory.mktemp("contract_index")))
    return open_store(pg_url())


@pytest.fixture(scope="module", params=["sqlite", "postgres"])
def any_store(request, filings_dir, tmp_path_factory):
    s = make_store(request.param, tmp_path_factory)
    ingest_folder(filings_dir, s, HashingEmbedder(), log=lambda m: None)
    return s


@pytest.fixture(scope="module")
def reference(filings_dir, tmp_path_factory):
    """A SQLite store with the same data, to compare against."""
    s = Store(str(tmp_path_factory.mktemp("reference_index")))
    ingest_folder(filings_dir, s, HashingEmbedder(), log=lambda m: None)
    return s


QUERIES = ["lithium battery supply shortage", "total net revenue", "frozen food distribution", "risk factors"]


# ------------------------------------------------------------ the same data

def test_counts_companies_and_years_match(any_store, reference):
    assert any_store.count() == reference.count() > 10
    assert any_store.companies() == reference.companies()
    assert any_store.years("ACME") == reference.years("ACME") == [2022, 2023]
    assert any_store.years("NOPE") == []
    assert any_store.embedder_name() == reference.embedder_name()


def test_same_ids_hold_the_same_passages(any_store, reference):
    ids = list(range(1, reference.count() + 1))
    assert [h.text for h in any_store.get(ids)] == [h.text for h in reference.get(ids)]


def test_get_keeps_order_and_skips_unknown_ids(any_store):
    hits = any_store.get([5, 999_999, 2, -1, 5])
    assert [h.id for h in hits] == [5, 2, 5]
    assert any_store.get([]) == []


def test_any_contains(any_store):
    assert any_store.any_contains("ACME", 2023, ["no such text", "1,250,400"])
    assert not any_store.any_contains("ACME", 2023, ["no such text"])
    assert not any_store.any_contains("NOPE", 2023, ["1,250,400"])
    assert not any_store.any_contains("ACME", 2023, [])


def test_reports_what_kind_of_store_it_is(any_store):
    assert any_store.kind in ("sqlite", "postgres")


# ------------------------------------------------------------ keyword search

def test_keyword_search_finds_revenue_and_respects_filters(any_store):
    hits = any_store.keyword_search("total net revenue", 5, ticker="ACME", year=2023)
    assert hits and all(h.ticker == "ACME" and h.fiscal_year == 2023 for h in hits)
    assert any("1,250,400" in h.text for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


@pytest.mark.parametrize("question", ["", "???", "the of and", "revenue AND (2023", 'NEAR("x"', "a-b:c*d",
                                      "'; DROP TABLE chunks; --", "revenue | !lithium & <->", "x" * 5000,
                                      "évolution du chiffre d'affaires", "收入"])
def test_keyword_search_survives_strange_questions(any_store, question):
    before = any_store.count()
    hits = any_store.keyword_search(question, 5)
    assert isinstance(hits, list) and len(hits) <= 5
    assert any_store.count() == before  # nothing was dropped or changed


def test_keyword_search_with_unknown_filters_is_empty(any_store):
    assert any_store.keyword_search("revenue", 5, ticker="NOPE") == []
    assert any_store.keyword_search("revenue", 5, year=1900) == []


# ------------------------------------------------------------ vector search

@pytest.mark.parametrize("question", QUERIES)
def test_dense_search_matches_the_reference(any_store, reference, question):
    q = HashingEmbedder().embed_query(question)
    got, want = any_store.dense_search(q, 8), reference.dense_search(q, 8)
    assert [h.score for h in got] == pytest.approx([h.score for h in want], abs=1e-4)
    for g, w in zip(got, want):
        # Same passage at each rank, except where two passages score the same
        # (then float rounding in either database may swap them).
        assert g.id == w.id or g.score == pytest.approx(w.score, abs=1e-6)


def test_dense_search_filters_return_every_matching_row(any_store, reference):
    # A vector index that filters after finding neighbours can quietly return
    # fewer rows than asked for. Asking for more than exist must give all of them.
    q = HashingEmbedder().embed_query("lithium battery supply shortage")
    bore_rows = len(reference.dense_search(q, 10_000, ticker="BORE"))
    hits = any_store.dense_search(q, 10_000, ticker="BORE")
    assert len(hits) == bore_rows > 0 and all(h.ticker == "BORE" for h in hits)
    acme_2022 = any_store.dense_search(q, 10_000, ticker="ACME", year=2022)
    assert acme_2022 and all((h.ticker, h.fiscal_year) == ("ACME", 2022) for h in acme_2022)


def test_dense_search_with_unknown_filters_is_empty(any_store):
    q = HashingEmbedder().embed_query("revenue")
    assert any_store.dense_search(q, 5, ticker="NOPE") == []
    assert any_store.dense_search(q, 5, year=1900) == []


def test_hit_ids_match_vectors(any_store):
    """Each passage's own vector must find that same passage first."""
    emb = HashingEmbedder()
    for h in any_store.get([1, 5, 9]):
        c = Chunk(h.ticker, h.company, h.fiscal_year, h.item, h.section, 0, h.kind, h.text)
        top = any_store.dense_search(emb.embed_passages([c.embed_text])[0], 1, ticker=h.ticker, year=h.fiscal_year)[0]
        assert top.id == h.id


def test_hybrid_search_works_end_to_end(any_store):
    hits, filters = Retriever(any_store, HashingEmbedder()).search("What was Acme's total net revenue in 2023?", k=5)
    assert filters.tickers == ["ACME"] and filters.year == 2023
    assert any("1,250,400" in h.text for h in hits)


# ------------------------------------------------------------ choosing a store and hiding passwords

def test_open_store_picks_sqlite_for_a_folder(tmp_path):
    s = open_store(str(tmp_path / "idx"))
    assert s.kind == "sqlite" and s.count() == 0


@pytest.mark.parametrize("url, hidden", [
    ("postgresql://filing:s3cret-pw@db.example:5432/filing", "s3cret-pw"),
    ("postgres://filing:s3cret-pw@db.example/filing", "s3cret-pw"),
    ("postgresql://db.example/filing?user=filing&password=s3cret-pw", "s3cret-pw"),
    ("postgresql://filing:p%40ss%3Aword@db.example/filing", "p%40ss%3Aword"),
])
def test_describe_location_hides_passwords(url, hidden):
    shown = describe_location(url)
    assert hidden not in shown and "db.example" in shown and "***" in shown


def test_describe_location_leaves_folders_alone():
    assert describe_location("index") == "index"


def test_a_failed_connection_does_not_leak_the_password():
    pytest.importorskip("psycopg")
    with pytest.raises(ConnectionError) as e:
        open_store("postgresql://filing:s3cret-pw@127.0.0.1:1/filing_test?connect_timeout=2")
    assert "s3cret-pw" not in str(e.value) and "127.0.0.1" in str(e.value)


def test_a_driver_error_that_repeats_the_password_is_cleaned():
    # Here the host name is the password, so the driver's own message contains it.
    pytest.importorskip("psycopg")
    with pytest.raises(ConnectionError) as e:
        open_store("postgresql://filing:s3cret-pw-host@s3cret-pw-host.invalid:5432/filing_test?connect_timeout=2")
    assert "s3cret-pw-host" not in str(e.value)


def test_the_api_works_on_either_store_and_never_shows_the_address(any_store):
    from fastapi.testclient import TestClient

    from filing_analyst.api import create_app

    c = TestClient(create_app(retriever=Retriever(any_store, HashingEmbedder())))
    health = c.get("/health")
    assert health.status_code == 200 and health.json()["store"] == any_store.kind
    assert "postgres" not in health.text.replace('"store":"postgres"', "")  # no address, user or password
    r = c.post("/search", json={"question": "What was Acme's total net revenue in 2023?", "k": 3})
    assert r.status_code == 200 and any("1,250,400" in p["text"] for p in r.json()["results"])
