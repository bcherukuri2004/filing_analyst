"""Container tests: build the image with docker compose, start it with the
real index mounted read-only, and check it from the outside.

Slow (it builds an image), so it only runs when asked:

    RUN_DOCKER_TESTS=1 pytest tests/test_docker.py -v

Needs Docker, and an index made by `python -m filing_analyst ingest data`.
Everything it starts is removed afterwards, pass or fail."""
import json
import os
import secrets
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "index")
PROJECT = "filing_analyst_test"
PORT = 8088  # not 8000, so it cannot clash with a server you are running
BASE = f"http://127.0.0.1:{PORT}"
KEY = secrets.token_urlsafe(24)  # a fresh throwaway key for each run

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


@pytest.fixture(scope="module")
def stack():
    if not shutil.which("docker"):
        pytest.skip("docker is not installed")
    if not os.path.exists(os.path.join(INDEX, "filings.db")):
        pytest.skip("no index; run: python -m filing_analyst ingest data")
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)  # never spend credits in tests
    env.update({"FILING_ANALYST_PORT": str(PORT), "FILING_ANALYST_INDEX_DIR": INDEX, "FILING_ANALYST_API_KEY": KEY})
    try:
        build = compose("build", env=env, check=False)
        assert build.returncode == 0, "image build failed:\n" + build.stderr[-3000:]
        up = compose("up", "-d", env=env, check=False)
        assert up.returncode == 0, "container did not start:\n" + up.stderr[-3000:]
        deadline = time.time() + 180
        while True:  # wait for the app to finish loading the models
            try:
                if http("GET", "/health")[0] == 200:
                    break
            except OSError:
                pass
            if time.time() > deadline:
                pytest.fail("no answer on /health after 3 minutes:\n" + compose("logs", env=env, check=False).stdout[-3000:])
            time.sleep(2)
        yield env
    finally:
        compose("down", "--volumes", "--remove-orphans", env=env, check=False)


def container_id(env) -> str:
    return compose("ps", "-q", "api", env=env).stdout.strip()


def exec_in(env, *cmd: str) -> subprocess.CompletedProcess:
    return compose("exec", "-T", "api", *cmd, env=env, check=False)


# ------------------------------------------------------------ it works

def test_health_reports_the_mounted_index(stack):
    from filing_analyst.store import Store

    code, body = http("GET", "/health")
    assert code == 200
    assert body["passages"] == Store(INDEX).count()
    assert body["auth_required"] is True and body["ask_enabled"] is False


def test_search_with_the_key_finds_nvidia(stack):
    code, body = http("POST", "/search", {"question": "What was NVIDIA total revenue for fiscal 2026?", "k": 3},
                      headers={"X-API-Key": KEY})
    assert code == 200
    assert body["filters"] == {"tickers": ["NVDA"], "year": 2026}
    assert body["results"] and all(r["ticker"] == "NVDA" for r in body["results"])


def test_the_key_passed_at_start_is_enforced(stack):
    assert http("POST", "/search", {"question": "NVIDIA revenue"})[0] == 401


def test_docker_reports_the_container_healthy(stack):
    deadline = time.time() + 120
    status = ""
    while time.time() < deadline:
        status = subprocess.run(["docker", "inspect", "--format", "{{.State.Health.Status}}", container_id(stack)],
                                capture_output=True, text=True).stdout.strip()
        if status == "healthy":
            break
        time.sleep(3)
    assert status == "healthy"


# ------------------------------------------------------------ it is locked down

def test_app_does_not_run_as_root(stack):
    r = exec_in(stack, "id", "-u")
    assert r.returncode == 0 and r.stdout.strip() != "0"


def test_index_is_read_only_inside_the_container(stack):
    # Ask Docker how the index is mounted. (A failed write alone would not prove
    # it: the app user may simply not own the folder.)
    mounts = json.loads(subprocess.run(["docker", "inspect", "--format", "{{json .Mounts}}", container_id(stack)],
                                       capture_output=True, text=True).stdout)
    index_mount = [m for m in mounts if m["Destination"] == "/data/index"]
    assert len(index_mount) == 1 and index_mount[0]["RW"] is False, index_mount
    # And a write really is refused, leaving nothing behind on this machine.
    r = exec_in(stack, "sh", "-c", "touch /data/index/should_not_exist")
    assert r.returncode != 0
    assert not os.path.exists(os.path.join(INDEX, "should_not_exist"))


def test_port_is_published_to_this_machine_only(stack):
    out = subprocess.run(["docker", "port", container_id(stack), "8000"], capture_output=True, text=True).stdout
    assert out.strip(), "port 8000 is not published"
    assert all(line.startswith("127.0.0.1:") for line in out.strip().splitlines()), out


def test_no_secrets_or_local_files_in_the_image(stack):
    image = subprocess.run(["docker", "inspect", "--format", "{{.Image}}", container_id(stack)],
                           capture_output=True, text=True).stdout.strip()
    env_vars = json.loads(subprocess.run(["docker", "image", "inspect", "--format", "{{json .Config.Env}}", image],
                                         capture_output=True, text=True).stdout)
    names = {e.split("=", 1)[0] for e in env_vars}
    assert not names & {"ANTHROPIC_API_KEY", "FILING_ANALYST_API_KEY", "SEC_USER_AGENT"}
    listing = exec_in(stack, "sh", "-c", "ls -a /app").stdout.split()
    for unwanted in (".venv", ".git", "data", "index", ".env"):
        assert unwanted not in listing, f"{unwanted} was copied into the image"


def test_models_are_in_the_image_and_work_offline(stack):
    # A new container with no network at all must still load and use both models.
    image = subprocess.run(["docker", "inspect", "--format", "{{.Image}}", container_id(stack)],
                           capture_output=True, text=True).stdout.strip()
    script = ("from filing_analyst.embed import FastEmbedder; from filing_analyst.retrieve import CrossEncoderReranker; "
              "v = FastEmbedder().embed_query('revenue'); s = CrossEncoderReranker().score('revenue', ['sales grew']); "
              "print(len(v), len(s))")
    r = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "python", image, "-c", script],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split()[-1] == "1"
