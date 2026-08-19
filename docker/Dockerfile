# syntax=docker/dockerfile:1
#
# Reproducible runtime image for the Scientific Discovery Engine.
#
# Base images are pinned by digest rather than by tag, because a tag is a
# mutable pointer: `python:3.11-slim-bookworm` resolves to a
# different image after every upstream security rebuild, so a build that is
# byte-identical in every other respect silently stops being reproducible. The
# digest is the image.
#
# Refreshing a pin is a deliberate act. Resolve the current digests with:
#
#     docker buildx imagetools inspect python:3.11-slim-bookworm
#     docker buildx imagetools inspect ghcr.io/astral-sh/uv:0.12.1
#
# Dependencies come from the committed uv.lock and nowhere else. `--frozen`
# makes the build fail rather than silently re-resolve when the lock and
# pyproject.toml disagree -- a build that quietly repairs a stale lock file
# produces an environment no one can reconstruct.

ARG PYTHON_IMAGE=python:3.11-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

# ---------------------------------------------------------------------------
# Builder: resolve the locked environment into a self-contained virtualenv.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.1@sha256:cf4eedcaa81655197f625739489effcbe71b61ceb1506f332c3facae5deceded /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Dependencies are installed before the source is copied so that a source-only
# change does not invalidate the dependency layer. --no-install-project is what
# makes that split possible.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY . .
# --no-editable is load-bearing, not a style choice: an editable install
# records a path reference back to /build, the source tree as it exists in
# *this* stage. The runtime stage below copies that tree to /app, a
# different path -- so an editable install's reference silently stops
# resolving, and `from cli.main import app` fails with "No module named
# 'cli'" despite the file being physically present. --no-editable copies
# the package into site-packages instead of referencing it by path, which
# is the only form of install that survives being copied to a new location.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---------------------------------------------------------------------------
# Runtime: the virtualenv and the source, without the build toolchain.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.title="Scientific Discovery Engine" \
      org.opencontainers.image.description="Domain-independent infrastructure for computational scientific discovery" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.base.name="python:3.11-slim-bookworm" \
      org.opencontainers.image.base.digest="sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3"

# Running as root is a default worth losing. The uid is fixed so that files
# written to a mounted volume have predictable ownership on the host.
RUN groupadd --system --gid 1001 sde \
    && useradd --system --uid 1001 --gid sde --create-home sde

WORKDIR /app

COPY --from=builder --chown=sde:sde /opt/venv /opt/venv
COPY --from=builder --chown=sde:sde /build /app

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER sde

# Fails the build if the locked environment cannot actually import the engine,
# which is cheaper to discover here than at run time.
RUN python -c "import engine; print('engine import ok')"

ENTRYPOINT ["sde"]
CMD ["--help"]
