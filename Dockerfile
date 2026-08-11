# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder
WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY engine ./engine
COPY algorithms ./algorithms
COPY validators ./validators
COPY plugins ./plugins
COPY cli ./cli
RUN pip install --no-cache-dir --prefix=/install ".[dev,gbm]"

FROM python:3.11-slim AS runtime
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY engine ./engine
COPY algorithms ./algorithms
COPY validators ./validators
COPY plugins ./plugins
COPY cli ./cli
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
# tests/test_domain_independence.py loads the checker from its path rather than
# importing it -- tools/ is not a package -- so the image needs it or that whole
# module fails to collect, taking 41 tests with it silently enough to look like
# a smaller suite rather than a broken one.
COPY tools ./tools

ENV SYMBOLIC_BACKEND=gplearn \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "physics_discovery.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
