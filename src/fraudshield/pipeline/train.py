"""
Stage 4: feature matrix -> trained model, threshold and test metrics.

    uv run python -m fraudshield.pipeline.train
"""

import json

import pandas as pd

from fraudshield.pipeline.config import ensure_parent, stage_args
from fraudshield.training.train import train


def main() -> None:
    cfg = stage_args("Train, tune threshold, evaluate on the test set.")
    paths = cfg["paths"]

    df = pd.read_csv(paths["features"], parse_dates=["timestamp"])
    result = train(df, model_dir=paths["model_dir"])

    report = {
        "best_model": result["best_model"],
        "threshold": result["threshold"],
        **result["metrics"],
    }
    with open(ensure_parent(paths["metrics"]), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
