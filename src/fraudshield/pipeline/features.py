"""
Stage 3: labelled transactions -> feature matrix.

    uv run python -m fraudshield.pipeline.features
"""

import pandas as pd

from fraudshield.features.engineer import build_features
from fraudshield.pipeline.config import ensure_parent, stage_args


def main() -> None:
    cfg = stage_args("Build the 28-feature matrix.")
    paths = cfg["paths"]

    df = pd.read_csv(paths["labelled"], parse_dates=["timestamp"])
    df = build_features(df)
    df.to_csv(ensure_parent(paths["features"]), index=False)


if __name__ == "__main__":
    main()
