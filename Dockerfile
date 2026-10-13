# FraudShield images, built from one Dockerfile with three stages:
#   base   -> shared Python runtime + system libraries
#   train  -> runs the full pipeline (make all) and writes models/ + reports/
#   serve  -> FastAPI app that loads a trained model from MODEL_DIR

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12.23

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------- base
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# libgomp1: OpenMP runtime needed by LightGBM and XGBoost (missing from -slim)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 app
WORKDIR /app

# ---------------------------------------------------------------- builders
# Dependencies are installed before the source is copied, so editing code
# does not invalidate the (slow) dependency layer.
FROM base AS build-common
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock ./

FROM build-common AS build-serve
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project --extra serve
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable --extra serve

FROM build-common AS build-train
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project --extra train
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable --extra train

# ---------------------------------------------------------------- train
FROM base AS train
RUN apt-get update \
 && apt-get install -y --no-install-recommends make \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build-train /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Makefile, config and src are needed because Make uses them as prerequisites
COPY Makefile ./
COPY configs ./configs
COPY src ./src
RUN mkdir -p data models reports && chown -R app:app /app
USER app
CMD ["make", "all", "PY=python -m"]

# ---------------------------------------------------------------- serve
FROM base AS serve
COPY --from=build-serve /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    MODEL_DIR=/app/models
RUN mkdir -p /app/models && chown app:app /app/models
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]
CMD ["uvicorn", "fraudshield.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
