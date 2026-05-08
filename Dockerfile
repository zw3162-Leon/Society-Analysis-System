FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY config.py main.py ./
COPY agents ./agents
COPY api ./api
COPY config ./config
COPY db ./db
COPY models ./models
COPY scripts ./scripts
COPY services ./services
COPY tools ./tools
COPY tests/fixtures ./tests/fixtures
COPY ui ./ui

RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

EXPOSE 8000 8501
