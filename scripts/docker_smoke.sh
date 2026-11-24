#!/usr/bin/env bash
# Smoke test for the Docker image: build it, start the whole stack (Postgres,
# the ingest step, the API) on a fresh empty database with three small
# FICTIONAL filings, and check that it answers. CI runs exactly this script,
# and it runs the same on a laptop:
#
#   scripts/docker_smoke.sh
#
# It uses throwaway passwords and its own project name, ports and volume, and
# removes everything it started when it ends, pass or fail. It never calls
# /ask, so it spends no API credits. Exit code 0 means every check passed.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT=fa_smoke
# Inside the project folder (not /tmp): Docker on a Mac can only see files under
# your home folder. Ignored by git and by docker build.
WORK=$(mktemp -d "$PWD/.smoke.XXXXXX")
export POSTGRES_PASSWORD="$(openssl rand -hex 16)"
export FILING_ANALYST_API_KEY="$(openssl rand -hex 24)"
export FILING_ANALYST_PORT="${SMOKE_PORT:-8097}" FILING_ANALYST_DB_PORT="${SMOKE_DB_PORT:-5546}"
export FILING_ANALYST_DATA_DIR="$WORK/filings" FILING_ANALYST_RERANK=0
unset ANTHROPIC_API_KEY
BASE="http://127.0.0.1:$FILING_ANALYST_PORT"

if docker compose version >/dev/null 2>&1; then COMPOSE=(docker compose); else COMPOSE=(docker-compose); fi
compose() { "${COMPOSE[@]}" -p "$PROJECT" "$@"; }

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "SMOKE TEST FAILED (exit $status). Last log lines:"
    compose logs --no-color --tail 60 || true
  fi
  compose down --volumes --remove-orphans >/dev/null 2>&1 || true
  rm -rf "$WORK"
  exit "$status"
}
trap cleanup EXIT

python3 scripts/write_sample_filings.py "$FILING_ANALYST_DATA_DIR"
compose down --volumes --remove-orphans >/dev/null 2>&1 || true

start=$(date +%s)
echo "== building the image and starting the stack"
compose up --build -d

echo "== waiting for /health"
for _ in $(seq 300); do
  if curl -sf -o /dev/null "$BASE/health"; then break; fi
  sleep 2
done
curl -sf -o /dev/null "$BASE/health" || { echo "no healthy /health after 10 minutes"; exit 1; }
echo "healthy after $(( $(date +%s) - start )) seconds (build included)"

echo "== checking the answers"
BASE="$BASE" python3 - <<'PY'
import json, os, sys, urllib.error, urllib.request

base, key = os.environ["BASE"], os.environ["FILING_ANALYST_API_KEY"]

def call(path, body=None, with_key=True):
    headers = {"Content-Type": "application/json"}
    if with_key:
        headers["X-API-Key"] = key
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body else None,
                                 method="POST" if body else "GET", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}

problems = []
def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what)
    if not ok:
        problems.append(what)

code, health = call("/health")
check(code == 200 and health.get("status") == "ok", "/health answers 200 and ok")
check(health.get("database") == "reachable" and health.get("store") == "postgres", "the database is reachable")
check((health.get("passages") or 0) > 0, f"the index has passages ({health.get('passages')})")
check(health.get("auth_required") is True and health.get("ask_enabled") is False, "a key is required; /ask is off")
code, live = call("/health/live")
check(code == 200 and live == {"status": "alive"}, "/health/live answers alive")
code, _ = call("/search", {"question": "What was Acme's total net revenue in 2023?"}, with_key=False)
check(code == 401, "/search without the key is refused (401)")
code, found = call("/search", {"question": "What was Acme's total net revenue in 2023?", "k": 3})
results = found.get("results", [])
check(code == 200 and results, "/search with the key returns passages")
check(any("1,250,400" in r["text"] for r in results), "the right passage (revenue 1,250,400) is in the top 3")
check(all(r["ticker"] == "ACME" and r["fiscal_year"] == 2023 for r in results), "results are Acme's 2023 filing")
sys.exit(1 if problems else 0)
PY

echo "== the API container runs as a normal user"
uid=$(compose exec -T api id -u)
[ "$uid" != "0" ] || { echo "the API runs as root"; exit 1; }
echo "  ok    user id $uid"

echo "SMOKE TEST PASSED in $(( $(date +%s) - start )) seconds"
