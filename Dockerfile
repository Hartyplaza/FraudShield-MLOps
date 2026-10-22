# FraudShield images, built from one Dockerfile with three stages:
#   base   -> shared Python runtime + system libraries
#   train  -> runs the DVC pipeline up to the quality gate, then copies
#             models/ and reports/ to /out (mounted volumes)
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
COPY --from=build-train /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    MLFLOW_DISABLE_AGENT_HINT=1
# The pipeline definition and everything its stages depend on
COPY configs ./configs
COPY src ./src
COPY dvc.yaml dvc.lock .dvcignore ./
COPY .dvc/config ./.dvc/config
# No Git inside the image, so DVC runs without it. Outputs are built inside the
# container and copied to /out at the end: DVC deletes a stage's outputs before
# re-running it, which would fail on a mounted volume.
RUN dvc config core.no_scm true \
 && mkdir -p /out/models /out/reports \
 && chown -R app:app /app /out
USER app
CMD ["sh", "-c", "dvc repro gate && cp -a models/. /out/models/ && cp -a reports/. /out/reports/"]

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
