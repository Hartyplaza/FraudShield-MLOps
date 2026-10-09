"""
Stage 2: cleaned transactions -> labelled transactions.

    uv run python -m fraudshield.pipeline.label
"""

import pandas as pd

from fraudshield.labelling.label_simulator import simulate_labels
from fraudshield.pipeline.config import ensure_parent, stage_args


def main() -> None:
    cfg = stage_args("Apply deterministic fraud labelling rules.")
    paths = cfg["paths"]

    df = pd.read_csv(paths["clean"], parse_dates=["timestamp"])
    df = simulate_labels(df)
    df.to_csv(ensure_parent(paths["labelled"]), index=False)


if __name__ == "__main__":
    main()
