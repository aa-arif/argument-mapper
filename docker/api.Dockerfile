# Extraction API.
#
# Installs only the `api` and `serve` extras: the training and evaluation
# dependencies (torch, transformers, datasets) are hundreds of megabytes and
# the service never imports them.

FROM python:3.11-slim AS base

# uv gives the same resolution as local development, from the same lockfile.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Dependencies first, so a source edit does not invalidate the install layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra api --extra serve

COPY src ./src
RUN uv sync --frozen --extra api --extra serve

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# One worker: the result cache is per-process, and a second worker would halve
# the hit rate for no throughput gain on a route whose latency is dominated by
# an upstream API call.
CMD ["uvicorn", "argmap.serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
