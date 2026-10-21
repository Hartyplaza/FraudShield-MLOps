"""
Stage 4: feature matrix -> trained model, threshold and test metrics.

The encoders and feature state from stage 3 are copied next to the model, so
models/ is a complete bundle: everything the API needs to score a raw
transaction.

    uv run python -m fraudshield.pipeline.train
"""

import json
import shutil
from pathlib import Path

import pandas as pd

from fraudshield.pipeline.config import ensure_parent, stage_args
from fraudshield.pipeline.tracking import log_run
from fraudshield.training.train import train


def main() -> None:
    cfg = stage_args("Train, tune threshold, evaluate on the test set.")
    paths = cfg["paths"]

    df = pd.read_csv(paths["features"], parse_dates=["timestamp"])
    result = train(df, model_dir=paths["model_dir"],
                   min_precision=cfg["training"]["min_precision"])

    model_dir = Path(paths["model_dir"])
    shutil.copy(paths["encoders"], model_dir / "encoders.json")
    shutil.copy(paths["feature_state"], model_dir / "feature_state.joblib")

    report = {
        "best_model": result["best_model"],
        "threshold": result["threshold"],
        **result["metrics"],
    }
    with open(ensure_parent(paths["metrics"]), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log_run(cfg, result, report)


if __name__ == "__main__":
    main()