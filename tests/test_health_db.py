"""Tests for the database check in /health and for reconnecting after the
database goes away.

Kubernetes (and anything else watching the service) asks /health "can you
serve right now?". If the database is down, the honest answer is a clear 503
saying so, not a crash. And when the database comes back, the service must
recover on its own, without a restart.

The first tests use a stand-in store that can be switched "down" on demand.
The last ones use a real Postgres and really cut the server's connection."""
import logging

import pytest
from fastapi.testclient import TestClient

from filing_analyst.api import create_app
from filing_analyst.embed import HashingEmbedder
from filing_analyst.ingest import ingest_folder
from filing_analyst.retrieve import Retriever


class Unreachable(Exception):
    """Plays the part of psycopg's "connection lost" error."""


class FlakyStore:
    """Wraps the real fixture store. While `down` is set, every database call
    fails the way a lost connection does. reconnect() brings it back only once
    `comes_back` is set, like a database that has restarted."""

    unavailable_errors = (Unreachable,)

    def __init__(self, inner):
        self.inner = inner
        self.kind = inner.kind
        self.location = inner.location
        self.down = False
        self.comes_back = False
        self.reconnects = 0

    def reconnect(self):
        self.reconnects += 1
        if not self.comes_back:
            raise ConnectionError("Could not connect to Postgres at postgresql://filing:***@db:5432/filing")
        self.down = False

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            if self.down:
                raise Unreachable("server closed the connection unexpectedly")
            return attr(*args, **kwargs)
        return call


@pytest.fixture
def flaky(store):
    s = FlakyStore(store)
    return s, TestClient(create_app(retriever=Retriever(s, HashingEmbedder())))


@pytest.fixture
def pg_client(filings_dir, tmp_path_factory):
    from test_store_contract import make_store

    pg = make_store("postgres", tmp_path_factory)
    ingest_folder(filings_dir, pg, HashingEmbedder(), log=lambda m: None)
    return pg, TestClient(create_app(retriever=Retriever(pg, HashingEmbedder())))


def search(client):
    return client.post("/search", json={"question": "What was Acme's total net revenue in 2023?"})


def test_health_says_the_database_is_reachable(retriever):
    body = TestClient(create_app(retriever=retriever)).get("/health").json()
    assert body["status"] == "ok" and body["database"] == "reachable"


def test_health_is_503_and_clear_while_the_database_is_down(flaky):
    store, client = flaky
    store.down = True
    r = client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "unavailable" and body["database"] == "unreachable"
    assert body["passages"] is None
    assert store.reconnects == 1  # it tried once to get the connection back before saying no


def test_search_is_503_not_a_crash_while_the_database_is_down(flaky):
    store, client = flaky
    store.down = True
    r = search(client)
    assert r.status_code == 503
    assert "database is unreachable" in r.json()["detail"]


def test_service_recovers_by_itself_when_the_database_comes_back(flaky):
    store, client = flaky
    assert search(client).status_code == 200
    store.down = True
    assert search(client).status_code == 503
    assert client.get("/health").status_code == 503
    store.comes_back = True  # the database has restarted; no restart of the service
    assert client.get("/health").json()["database"] == "reachable"
    r = search(client)
    assert r.status_code == 200 and r.json()["results"]


def test_a_dropped_connection_is_retried_inside_the_same_request(flaky):
    # The connection broke between two requests; the database itself is fine.
    # The caller should never see that: one reconnect, then a normal answer.
    store, client = flaky
    store.down, store.comes_back = True, True
    r = search(client)
    assert r.status_code == 200 and r.json()["results"]
    assert store.reconnects == 1


def test_the_outage_never_leaks_the_password_or_error_text(flaky, caplog):
    store, client = flaky
    store.down = True
    with caplog.at_level(logging.WARNING):
        texts = [client.get("/health").text, search(client).text]
    for text in texts:
        assert "server closed the connection" not in text  # raw driver errors stay out of replies
    for text in texts + [caplog.text]:
        assert "filing:" not in text  # no user, no password, no address


def test_other_errors_are_not_hidden_as_a_database_outage(store):
    # Only "lost connection" errors mean the database is down. A bug elsewhere
    # must still show up as a real server error, not a reassuring 503.
    class Broken(FlakyStore):
        def count(self):
            raise ZeroDivisionError("a real bug")

    s = Broken(store)
    client = TestClient(create_app(retriever=Retriever(s, HashingEmbedder())), raise_server_exceptions=False)
    assert client.get("/health").status_code == 500
    assert s.reconnects == 0


def test_sqlite_ping_and_reconnect_are_harmless(store):
    store.ping()
    store.reconnect()
    assert store.count() > 0
    assert store.unavailable_errors == ()


# ------------------------------------------------------------ a real Postgres

def test_postgres_recovers_after_its_connection_is_cut(pg_client):
    """Really cut the server's connection from a second session (what a
    database restart or a network blip does), then check the very next
    request still works."""
    import psycopg

    from test_store_contract import pg_url

    pg, client = pg_client
    assert search(client).status_code == 200
    first_pid = pg.db.info.backend_pid
    with psycopg.connect(pg_url(), autocommit=True) as admin:
        admin.execute("SELECT pg_terminate_backend(%s)", (first_pid,))
    r = search(client)
    assert r.status_code == 200 and r.json()["results"]
    assert pg.db.info.backend_pid != first_pid  # a new connection was made
    # The vector search settings belong to a connection, so the new one must get them again.
    assert pg.db.execute("SHOW hnsw.iterative_scan").fetchone()["hnsw.iterative_scan"] == "relaxed_order"
    assert pg.db.execute("SHOW hnsw.ef_search").fetchone()["hnsw.ef_search"] == str(pg.EF_SEARCH)
    assert client.get("/health").json()["database"] == "reachable"
    pg.ping()


def test_postgres_that_stays_down_gives_503_without_the_password(pg_client):
    import psycopg

    from test_store_contract import pg_url

    pg, client = pg_client
    with psycopg.connect(pg_url(), autocommit=True) as admin:
        admin.execute("SELECT pg_terminate_backend(%s)", (pg.db.info.backend_pid,))
    # Point reconnects at a port where nothing listens, with a recognisable password.
    pg._url = "postgresql://filing:Sup3rSecretPw@127.0.0.1:1/filing_test?connect_timeout=2"
    for r in (client.get("/health"), search(client)):
        assert r.status_code == 503
        assert "Sup3rSecretPw" not in r.text
    with pytest.raises(pg.unavailable_errors):
        pg.ping()
    with pytest.raises(ConnectionError) as e:
        pg.reconnect()
    assert "Sup3rSecretPw" not in str(e.value) and "***" in str(e.value)
