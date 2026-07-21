# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder
WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY physics_discovery ./physics_discovery
COPY engine ./engine
COPY cli ./cli
RUN pip install --no-cache-dir --prefix=/install ".[dev,gbm]"

FROM python:3.11-slim AS runtime
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY physics_discovery ./physics_discovery
COPY engine ./engine
COPY cli ./cli
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests

ENV SYMBOLIC_BACKEND=gplearn \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "physics_discovery.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
