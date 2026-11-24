"""Container tests: start the whole stack with docker compose (Postgres with
pgvector, the one time ingest step and the API) on a fresh, empty database
volume, and check it from the outside.

Slow (it builds the image and the index), so it only runs when asked:

    RUN_DOCKER_TESTS=1 pytest tests/test_docker.py -v

Needs Docker and the downloaded filings in ./data. It uses its own project
name, ports and volume, so it never touches a stack you are running, and
everything it starts is removed afterwards, pass or fail."""
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PROJECT = "filing_analyst_test"
PORT = 8088     # not 8000, so it cannot clash with a server you are running
DB_PORT = 5544  # not 5433, for the same reason
BASE = f"http://127.0.0.1:{PORT}"
KEY = secrets.token_urlsafe(24)       # a fresh throwaway key for each run
DB_PASSWORD = secrets.token_hex(16)   # and a fresh throwaway database password

pytestmark = pytest.mark.skipif(os.environ.get("RUN_DOCKER_TESTS") != "1",
                                reason="slow; set RUN_DOCKER_TESTS=1 to build and test the container")


def compose_cmd() -> list[str]:
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    return ["docker", "compose"]


def compose(*args: str, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([*compose_cmd(), "-p", PROJECT, *args], cwd=ROOT, env=env or os.environ.copy(),
                          capture_output=True, text=True, check=check, timeout=1800)


def http(method: str, path: str, body: dict | None = None, headers: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(BASE + path, method=method, headers={"Content-Type": "application/json", **(headers or {})},
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def wait_for_health(env, seconds: int) -> None:
    deadline = time.time() + seconds
    while True:
        try:
            if http("GET", "/health")[0] == 200:
                return
        except OSError:
            pass
        if time.time() > deadline:
            pytest.fail(f"no healthy answer on /health after {seconds}s:\n"
                        + compose("logs", env=env, check=False).stdout[-3000:])
        time.sleep(2)


def has_filings() -> bool:
    return os.path.isdir(DATA) and any(f.endswith(".htm") for f in os.listdir(DATA))


def expected_passages() -> int:
    """Index the same filings into a scratch SQLite store, as the reference count."""
    from filing_analyst.ingest import ingest_folder
    from filing_analyst.store import Store

    class Ones:  # the count does not depend on the vectors, so skip the real model
        name = "ones"

        def embed_passages(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32)

    with tempfile.TemporaryDirectory() as d:
        return ingest_folder(DATA, Store(d), Ones(), log=lambda m: None)


@pytest.fixture(scope="module")
def stack():
    if not shutil.which("docker"):
        pytest.skip("docker is not installed")
    if not has_filings():
        pytest.skip("no filings in ./data; run scripts/download_filings.py first")
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)  # never spend credits in tests
    env.update({"FILING_ANALYST_PORT": str(PORT), "FILING_ANALYST_DB_PORT": str(DB_PORT),
                "FILING_ANALYST_API_KEY": KEY, "POSTGRES_PASSWORD": DB_PASSWORD, "FILING_ANALYST_DATA_DIR": DATA})
    compose("down", "--volumes", "--remove-orphans", env=env, check=False)  # start from an empty database
    try:
        build = compose("build", env=env, check=False)
        assert build.returncode == 0, "image build failed:\n" + build.stderr[-3000:]
        up = compose("up", "-d", env=env, check=False)
        assert up.returncode == 0, ("the stack did not start:\n" + up.stderr[-3000:]
                                    + compose("logs", env=env, check=False).stdout[-3000:])
        wait_for_health(env, 600)  # the first start builds the index
        yield env
    finally:
        compose("down", "--volumes", "--remove-orphans", env=env, check=False)


def container_id(env, service: str = "api") -> str:
    return compose("ps", "-a", "-q", service, env=env).stdout.strip()


def inspect(cid: str, fmt: str) -> str:
    return subprocess.run(["docker", "inspect", "--format", fmt, cid], capture_output=True, text=True).stdout.strip()


def exec_in(env, *cmd: str) -> subprocess.CompletedProcess:
    return compose("exec", "-T", "api", *cmd, env=env, check=False)


# ------------------------------------------------------------ it works

def test_health_reports_the_whole_index_in_postgres(stack):
    code, body = http("GET", "/health")
    assert code == 200
    assert body["store"] == "postgres" and body["database"] == "reachable"
    assert body["passages"] == expected_passages()
    assert body["auth_required"] is True and body["ask_enabled"] is False


def test_search_with_the_key_finds_nvidia(stack):
    code, body = http("POST", "/search", {"question": "What was NVIDIA total revenue for fiscal 2026?", "k": 3},
                      headers={"X-API-Key": KEY})
    assert code == 200
    assert body["filters"] == {"tickers": ["NVDA"], "year": 2026}
    assert body["results"] and all(r["ticker"] == "NVDA" for r in body["results"])
    assert all({"company", "fiscal_year", "item"} <= set(r) for r in body["results"])


def test_the_key_passed_at_start_is_enforced(stack):
    assert http("POST", "/search", {"question": "NVIDIA revenue"})[0] == 401


def test_the_ingest_step_built_the_index_and_exited(stack):
    cid = container_id(stack, "ingest")
    assert inspect(cid, "{{.State.Status}}") == "exited"
    assert inspect(cid, "{{.State.ExitCode}}") == "0"
    assert "Indexed" in compose("logs", "ingest", env=stack, check=False).stdout


def test_docker_reports_the_container_healthy(stack):
    deadline = time.time() + 120
    status = ""
    while time.time() < deadline:
        status = inspect(container_id(stack), "{{.State.Health.Status}}")
        if status == "healthy":
            break
        time.sleep(3)
    assert status == "healthy"


# ------------------------------------------------------------ it is locked down

def test_compose_refuses_to_start_without_a_database_password():
    env = {k: v for k, v in os.environ.items() if k != "POSTGRES_PASSWORD"}
    r = subprocess.run([*compose_cmd(), "-p", PROJECT + "_nopw", "config"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "POSTGRES_PASSWORD" in r.stderr


def test_app_does_not_run_as_root(stack):
    r = exec_in(stack, "id", "-u")
    assert r.returncode == 0 and r.stdout.strip() != "0"


def test_the_filings_are_read_only_and_the_api_has_none(stack):
    mounts = json.loads(inspect(container_id(stack, "ingest"), "{{json .Mounts}}"))
    data = [m for m in mounts if m["Destination"] == "/data/filings"]
    assert len(data) == 1 and data[0]["RW"] is False, data
    assert json.loads(inspect(container_id(stack), "{{json .Mounts}}")) == []  # the API needs no files at all


def test_the_password_is_not_in_any_address_or_log(stack):
    assert DB_PASSWORD not in compose("logs", env=stack, check=False).stdout
    assert DB_PASSWORD not in exec_in(stack, "sh", "-c", "echo $FILING_ANALYST_INDEX").stdout


def test_ports_are_published_to_this_machine_only(stack):
    for service, port in (("api", "8000"), ("db", "5432")):
        out = subprocess.run(["docker", "port", container_id(stack, service), port],
                             capture_output=True, text=True).stdout
        assert out.strip(), f"{service} port {port} is not published"
        assert all(line.startswith("127.0.0.1:") for line in out.strip().splitlines()), out


def test_no_secrets_or_local_files_in_the_image(stack):
    image = inspect(container_id(stack), "{{.Image}}")
    env_vars = json.loads(subprocess.run(["docker", "image", "inspect", "--format", "{{json .Config.Env}}", image],
                                         capture_output=True, text=True).stdout)
    names = {e.split("=", 1)[0] for e in env_vars}
    assert not names & {"ANTHROPIC_API_KEY", "FILING_ANALYST_API_KEY", "SEC_USER_AGENT", "PGPASSWORD",
                        "POSTGRES_PASSWORD"}
    listing = exec_in(stack, "sh", "-c", "ls -a /app").stdout.split()
    for unwanted in (".venv", ".git", "data", "index", ".env"):
        assert unwanted not in listing, f"{unwanted} was copied into the image"


def test_models_are_in_the_image_and_work_offline(stack):
    # A new container with no network at all must still load and use both models.
    image = inspect(container_id(stack), "{{.Image}}")
    script = ("from filing_analyst.embed import FastEmbedder; from filing_analyst.retrieve import CrossEncoderReranker; "
              "v = FastEmbedder().embed_query('revenue'); s = CrossEncoderReranker().score('revenue', ['sales grew']); "
              "print(len(v), len(s))")
    r = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "python", image, "-c", script],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split()[-1] == "1"


# Last on purpose: it stops and restarts the stack the tests above share.
def test_zz_a_restart_keeps_the_index_and_skips_the_rebuild(stack):
    count = http("GET", "/health")[1]["passages"]
    compose("stop", env=stack)
    up = compose("up", "-d", env=stack, check=False)
    assert up.returncode == 0, up.stderr[-3000:]
    wait_for_health(stack, 300)
    assert "already has" in compose("logs", "--tail", "5", "ingest", env=stack, check=False).stdout
    assert http("GET", "/health")[1]["passages"] == count
