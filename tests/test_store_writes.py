"""Writing to a store, on both SQLite and Postgres: replacing the index, bad
input refused without losing the old index, and searches over more rows than
a vector index looks at by default.

Each test gets an empty store of its own. For Postgres that is the test
database from FILING_ANALYST_TEST_PG, which these tests overwrite."""
import numpy as np
import pytest
from test_store_contract import make_store, pg_url

from filing_analyst.chunk import Chunk


@pytest.fixture(params=["sqlite", "postgres"])
def fresh(request, tmp_path_factory):
    if request.param == "postgres":
        import psycopg

        with psycopg.connect(pg_url(), autocommit=True) as conn:  # start from an empty database
            conn.execute("DROP TABLE IF EXISTS chunks, meta")
    return make_store(request.param, tmp_path_factory)


def chunk(n: int, text: str = "", ticker: str = "TEST", year: int = 2024) -> Chunk:
    return Chunk(ticker, "Test Co", year, "7", "MD&A", n, "text", text or f"passage number {n} about revenue")


def unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_replace_all_replaces_instead_of_adding(fresh):
    fresh.replace_all([chunk(i) for i in range(5)], unit_vectors(5, 8), "e1")
    fresh.replace_all([chunk(i) for i in range(3)], unit_vectors(3, 8, seed=1), "e2")
    assert fresh.count() == 3
    assert fresh.embedder_name() == "e2"
    assert [h.id for h in fresh.get([1, 2, 3, 4])] == [1, 2, 3]


def test_a_new_index_may_use_a_different_vector_size(fresh):
    # Switching embedding models changes the size of every vector.
    fresh.replace_all([chunk(i) for i in range(4)], unit_vectors(4, 16), "small")
    vecs = unit_vectors(4, 64, seed=2)
    fresh.replace_all([chunk(i) for i in range(4)], vecs, "big")
    assert fresh.dense_search(vecs[2], 1)[0].id == 3


@pytest.mark.parametrize("bad", ["nan", "inf", "count"])
def test_bad_input_is_refused_and_the_old_index_is_kept(fresh, bad):
    fresh.replace_all([chunk(i) for i in range(4)], unit_vectors(4, 8), "good")
    vecs = unit_vectors(4, 8, seed=3)
    chunks = [chunk(i, "new text") for i in range(4)]
    if bad == "nan":
        vecs[2, 1] = np.nan
    elif bad == "inf":
        vecs[0, 0] = np.inf
    else:
        chunks = chunks[:3]  # one vector too many
    with pytest.raises(ValueError):
        fresh.replace_all(chunks, vecs, "bad")
    assert fresh.count() == 4 and fresh.embedder_name() == "good"
    assert "new text" not in fresh.get([1])[0].text


def test_a_failure_halfway_through_keeps_the_old_index(fresh):
    fresh.replace_all([chunk(i) for i in range(4)], unit_vectors(4, 8), "good")
    rows = [chunk(i) for i in range(4)]
    rows[3] = Chunk("TEST", "Test Co", 2024, "7", "MD&A", 3, "text", None)  # no text: fails on the last insert
    with pytest.raises(Exception):
        fresh.replace_all(rows, unit_vectors(4, 8, seed=4), "broken")
    assert fresh.count() == 4 and fresh.embedder_name() == "good"


def test_text_with_odd_characters_round_trips(fresh):
    # Postgres cannot store the NUL character in text, so both stores drop it.
    texts = ["nul\x00inside", "quote ' and \" and \\ backslash", "emoji 📈 and é", "line\nbreak\ttab", "%s %(x)s ?"]
    fresh.replace_all([chunk(i, t) for i, t in enumerate(texts)], unit_vectors(5, 8), "e")
    got = [h.text for h in fresh.get([1, 2, 3, 4, 5])]
    assert got == [t.replace("\x00", "") for t in texts]
    assert fresh.any_contains("TEST", 2024, ["📈"])
    assert fresh.any_contains("TEST", 2024, ["%s %(x)s ?"])


def test_vector_search_returns_k_rows_beyond_the_index_default(fresh):
    # pgvector's index looks at 40 candidates by default. The retriever asks
    # for 50, so a store that kept the default would quietly return 40.
    n = 300
    vecs = unit_vectors(n, 32, seed=5)
    fresh.replace_all([chunk(i, ticker="A" if i % 3 else "B") for i in range(n)], vecs, "e")
    q = unit_vectors(1, 32, seed=6)[0]
    assert len(fresh.dense_search(q, 50)) == 50
    assert len(fresh.dense_search(q, 250)) == 250
    assert len(fresh.dense_search(q, 10_000)) == n
    only_b = fresh.dense_search(q, 10_000, ticker="B")
    assert len(only_b) == n // 3 and all(h.ticker == "B" for h in only_b)


def test_vector_search_order_is_exact(fresh):
    n = 300
    vecs = unit_vectors(n, 32, seed=7)
    fresh.replace_all([chunk(i) for i in range(n)], vecs, "e")
    q = unit_vectors(1, 32, seed=8)[0]
    want = list(np.argsort(-(vecs @ q))[:20] + 1)
    got = [h.id for h in fresh.dense_search(q, 20)]
    assert got == want
    scores = [h.score for h in fresh.dense_search(q, 20)]
    assert scores == sorted(scores, reverse=True)


def test_an_empty_store_answers_without_errors(fresh):
    assert fresh.count() == 0
    assert fresh.companies() == [] and fresh.years("A") == [] and fresh.get([1]) == []
    assert fresh.keyword_search("revenue", 5) == []
    assert not fresh.any_contains("A", 2024, ["x"])


@pytest.fixture
def big_pg(tmp_path_factory):
    """3,000 passages in Postgres, with Postgres told to always use the vector
    index. (On small tables it rightly reads every row instead, which would
    hide any mistake in how the index is used.)"""
    store = make_store("postgres", tmp_path_factory)
    n = 3000
    vecs = unit_vectors(n, 32, seed=9)
    store.replace_all([chunk(i, ticker="A" if i % 3 else "B") for i in range(n)], vecs, "e")
    store.db.execute("SET enable_seqscan = off")
    return store, vecs


def test_postgres_really_uses_the_vector_index(big_pg):
    store, _ = big_pg
    q = "[" + ",".join(["0.1"] * 32) + "]"
    plan = " ".join(r["QUERY PLAN"] for r in store.db.execute(
        "EXPLAIN SELECT id FROM chunks ORDER BY embedding <#> %s::vector LIMIT 50", (q,)))
    assert "chunks_embedding" in plan, plan


def test_index_search_returns_every_row_asked_for(big_pg):
    store, vecs = big_pg
    q = unit_vectors(1, 32, seed=10)[0]
    assert len(store.dense_search(q, 50)) == 50      # more than pgvector's default of 40
    assert len(store.dense_search(q, 500)) == 500    # more than our candidate list of 200
    only_b = store.dense_search(q, 2000, ticker="B")  # a filter removes two thirds of candidates
    assert len(only_b) == 1000 and all(h.ticker == "B" for h in only_b)


def test_index_search_results_are_sorted_and_nearly_exact(big_pg):
    store, vecs = big_pg
    q = unit_vectors(1, 32, seed=11)[0]
    hits = store.dense_search(q, 50)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    exact = set(np.argsort(-(vecs @ q))[:50] + 1)
    found = len(exact & {h.id for h in hits})
    assert found >= 45, f"only {found} of the true 50 nearest found"  # HNSW is approximate, but close


def test_equal_scores_come_back_in_id_order(fresh):
    # The same question must always give the same answer, even when scores tie.
    v = unit_vectors(1, 8)[0]
    fresh.replace_all([chunk(i) for i in range(30)], np.tile(v, (30, 1)), "e")
    assert [h.id for h in fresh.dense_search(v, 10)] == list(range(1, 11))
