# syntax=docker/dockerfile:1.7

# ---- build: resolve dependencies with uv into /app/.venv -------------------------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
# Optional extras, e.g. EXTRAS=rerank for the local cross-encoder (ONNX Runtime + model:
# the image grows from ~0.4 to ~0.85 GB uncompressed).
ARG EXTRAS=""
ARG RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project ${EXTRAS:+--extra $EXTRAS}

COPY alembic.ini ./
COPY migrations ./migrations
COPY scripts ./scripts
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev ${EXTRAS:+--extra $EXTRAS}

# Bake the tokenizer (and, with the rerank extra, the cross-encoder) into the image so
# containers never need to download them.
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
RUN if [ "$EXTRAS" = "rerank" ]; then /app/.venv/bin/python -c \
    "from fastembed.rerank.cross_encoder import TextCrossEncoder as T; T('$RERANK_MODEL', cache_dir='/app/.models')"; fi

# ---- runtime -------------------------------------------------------------------
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 docuchat \
    && mkdir -p /data/uploads && chown docuchat:docuchat /data/uploads
WORKDIR /app
COPY --from=builder --chown=docuchat:docuchat /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken \
    RERANK_CACHE_DIR=/app/.models \
    UPLOAD_DIR=/data/uploads \
    LOG_FORMAT=json

USER docuchat
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# Apply migrations, then serve.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --proxy-headers"]
