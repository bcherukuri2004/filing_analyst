#!/bin/bash
# Runs once, as root, when the EC2 server first starts (EC2 "user data").
# NOT RUN YET: part of docs/aws-plan.md, waiting for approval.
#
# Before launch, two placeholders are filled in on your Mac (see the plan):
#   __BRANCH__          the git branch to run
#   __SEC_USER_AGENT__  "Your Name you@example.com", which the SEC asks for
# The database password and the API key are made here, on the server, and
# never leave it. Progress goes to /var/log/filing-analyst-setup.log.
set -euo pipefail
exec > /var/log/filing-analyst-setup.log 2>&1

dnf install -y docker git
systemctl enable --now docker
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-compose \
  https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-aarch64
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# A 2 GB swap file: a safety net for the 2 GB server while the index is built.
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo "/swapfile none swap defaults 0 0" >> /etc/fstab

git clone --depth 1 --branch "__BRANCH__" https://github.com/bcherukuri2004/filing_analyst /opt/filing_analyst
cd /opt/filing_analyst

umask 077  # .env readable by root only
{
  echo "POSTGRES_PASSWORD=$(openssl rand -hex 24)"
  echo "FILING_ANALYST_API_KEY=$(openssl rand -hex 24)"
} > .env

SEC_USER_AGENT="__SEC_USER_AGENT__" python3 scripts/download_filings.py data AAPL MSFT NVDA GOOGL AMZN --years 2

date +"setup: starting docker compose at %T"
docker compose up --build -d
date +"setup: docker compose started at %T"
