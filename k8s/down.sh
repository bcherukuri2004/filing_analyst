#!/usr/bin/env bash
# Delete the local cluster and everything in it (the index too).
set -euo pipefail
kind delete cluster --name filing-analyst
