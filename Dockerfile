# FraudShield images, built from one Dockerfile:
#   base   -> shared Python runtime + system libraries
#   train  -> runs the DVC pipeline up to the quality gate, then copies
#             models/ and reports/ to /out (mounted volumes)
#   serve  -> FastAPI app that loads a trained model from MODEL_DIR
#   app    -> serve + a model trained and quality-gated during the build:
#             one self-contained image for deployment (no volumes needed)

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
CMD ["sh", "-c", "dvc repro gate reference && cp -a models/. /out/models/ && cp -a reports/. /out/reports/"]

# ---------------------------------------------------------------- serve
FROM base AS serve
COPY --from=build-serve /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    MODEL_DIR=/app/models \
    REFERENCE_PATH=/app/reports/reference.json
RUN mkdir -p /app/models && chown app:app /app/models
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.getenv('PORT', '8000'), timeout=2)"
# Cloud hosts (Cloud Run, Render) tell the app which port to use via $PORT
CMD ["sh", "-c", "exec uvicorn fraudshield.serving.api:app --host 0.0.0.0 --port ${PORT:-8000}"]

# ---------------------------------------------------------------- trained
# Runs the pipeline during the build. If the quality gate fails, the build fails,
# so a model that did not pass the gate can never end up in a deployable image.
FROM train AS trained
RUN dvc repro gate reference

# ---------------------------------------------------------------- app
# The deployable image: the API plus the gated model, nothing to mount.
FROM serve AS app
COPY --from=trained --chown=app:app /app/models /app/models
COPY --from=trained --chown=app:app /app/reports /app/reports
# The commit this image was built from; /health reports it as "revision".
# Last in the file, so changing it never invalidates the cached layers above.
ARG REVISION=unknown
ENV APP_REVISION=$REVISION
