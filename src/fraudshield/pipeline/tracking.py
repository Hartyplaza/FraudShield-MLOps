"""
src/fraudshield/pipeline/tracking.py

Log one training run to MLflow: what went in (code version, data hashes,
config), what came out (metrics, the model bundle), and how it compares with
the other candidate models.

Tracking is local by default (configs/pipeline.yaml -> tracking.uri). The
MLFLOW_TRACKING_URI environment variable overrides it, e.g. to point at a
shared server later. Name a run with RUN_NAME=<name> make train.

Browse runs:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_dirty() -> str:
    """Whether the run used uncommitted code. MLflow already records the commit."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return str(bool(out.strip())).lower()


def log_run(cfg: dict, result: dict, report: dict) -> str | None:
    """Log a finished training run. Returns the run id, or None if disabled."""
    tracking = cfg.get("tracking", {})
    if not tracking.get("enabled", True):
        return None
    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri(tracking.get("uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment(tracking.get("experiment", "fraudshield"))

    paths, gen = cfg["paths"], cfg["generator"]
    model_dir = Path(paths["model_dir"])
    feature_cols = json.loads((model_dir / "feature_cols.json").read_text())

    with mlflow.start_run(run_name=os.environ.get("RUN_NAME")) as run:
        mlflow.set_tags({
            "git_dirty": git_dirty(),
            "best_model": result["best_model"],
            "data.raw_sha256": file_sha256(paths["raw"]),
            "data.features_sha256": file_sha256(paths["features"]),
        })
        mlflow.log_params({
            "data.seed": gen["seed"],
            "data.n_rows": gen["n_rows"],
            "data.days": gen["days"],
            "n_features": len(feature_cols),
            "threshold": result["threshold"],
            **{f"best.{k}": v for k, v in result["best_params"].items()},
        })
        mlflow.log_metrics({
            **{f"test_{k}": float(v) for k, v in result["metrics"].items()},
            **{f"cv_pr_auc.{name}": s for name, s in result["cv_scores"].items()},
        })
        mlflow.log_dict(report, "metrics.json")
        mlflow.log_artifacts(str(model_dir), artifact_path="model_bundle")
        logger.info("MLflow run %s logged to experiment '%s'",
                    run.info.run_id, tracking.get("experiment", "fraudshield"))
        return run.info.run_id