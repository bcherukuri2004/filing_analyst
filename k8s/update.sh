#!/usr/bin/env bash
# Rolling update of the API to a new image version, for example:
#   k8s/update.sh v2
# Builds filing-analyst:<version>, copies it into the cluster and replaces the
# pods one at a time; there are always 2 ready copies serving.
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION="${1:?give a version, for example: k8s/update.sh v2}"
K="kubectl --context kind-filing-analyst -n filing-analyst"
docker build -q -t "filing-analyst:$VERSION" --build-arg FILING_ANALYST_VERSION="$VERSION" . >/dev/null
kind load docker-image "filing-analyst:$VERSION" --name filing-analyst >/dev/null
$K set image deployment/api api="filing-analyst:$VERSION"
$K rollout status deployment/api --timeout=300s
