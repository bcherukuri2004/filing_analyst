#!/usr/bin/env bash
# Start the whole service on a local Kubernetes cluster made with kind.
#
#   k8s/up.sh          build image v1, make the cluster, start everything
#   k8s/down.sh        delete the cluster and everything in it
#
# Safe to run again: each step skips what already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
K="kubectl --context kind-filing-analyst -n filing-analyst"

echo "== building image filing-analyst:v1"
docker build -q -t filing-analyst:v1 --build-arg FILING_ANALYST_VERSION=v1 . >/dev/null

if ! kind get clusters 2>/dev/null | grep -qx filing-analyst; then
  echo "== creating the cluster"
  sed "s#FILINGS_DIR#$PWD/data#" k8s/kind-config.yaml | kind create cluster --config -
fi

echo "== copying the image into the cluster"
kind load docker-image filing-analyst:v1 --name filing-analyst >/dev/null

kubectl --context kind-filing-analyst apply -f k8s/namespace.yaml
$K apply -f k8s/configmap.yaml
if ! $K get secret filing-analyst-secrets >/dev/null 2>&1; then
  echo "== creating the Secret with fresh random values (never written to a file)"
  $K create secret generic filing-analyst-secrets \
    --from-literal=PGPASSWORD="$(openssl rand -hex 24)" \
    --from-literal=FILING_ANALYST_API_KEY="$(openssl rand -hex 24)"
fi

echo "== starting Postgres"
$K apply -f k8s/postgres.yaml
$K rollout status statefulset/postgres --timeout=300s

echo "== building the index (only the first time)"
$K apply -f k8s/ingest-job.yaml
$K wait --for=condition=complete job/ingest --timeout=1800s

echo "== starting the API (2 replicas)"
$K apply -f k8s/app.yaml
$K rollout status deployment/api --timeout=300s
$K get pods -o wide
echo "Ready: curl http://127.0.0.1:8080/health"
