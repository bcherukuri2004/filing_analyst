# filing_analyst web API in a container.
#
#   docker compose up --build        (see docker-compose.yml and the README)
#
# The image holds the code, its Python packages and the two models. It does
# NOT hold the search index (mounted read-only at run time) or any keys
# (passed in as environment variables at run time).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/opt/models \
    FILING_ANALYST_INDEX=/data/index \
    HOME=/tmp

WORKDIR /app

# Install the packages first, from pyproject.toml alone, so this slow layer is
# reused when only the code changes.
COPY pyproject.toml README.md ./
RUN mkdir filing_analyst && touch filing_analyst/__init__.py \
 && pip install ".[api]" \
 && pip uninstall -y filing_analyst && rm -rf filing_analyst build *.egg-info

# Download both models now, while building, so the container starts without
# internet and without a 150 MB download on every start.
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')" \
 && chmod -R a+rX /opt/models

COPY filing_analyst ./filing_analyst
RUN pip install --no-deps . && rm -rf build *.egg-info

# From here on, never look for model updates online.
ENV HF_HUB_OFFLINE=1

# Run as an ordinary user, not root.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER app

EXPOSE 8000

# Docker marks the container unhealthy if /health stops answering.
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# 0.0.0.0 inside the container is needed for the port to be reachable at all.
# Who can reach it is decided by the port mapping in docker-compose.yml.
CMD ["uvicorn", "filing_analyst.api:app", "--host", "0.0.0.0", "--port", "8000"]
