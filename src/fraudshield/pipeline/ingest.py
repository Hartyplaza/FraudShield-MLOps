"""
Stage 1: raw logs -> parsed and cleaned transactions.

    uv run python -m fraudshield.pipeline.ingest
"""

import pandas as pd

from fraudshield.ingestion.cleaner import clean
from fraudshield.ingestion.parser import parse_logs
from fraudshield.pipeline.config import ensure_parent, stage_args


def main() -> None:
    cfg = stage_args("Parse and clean raw transaction logs.")
    paths = cfg["paths"]

    raw = pd.read_csv(paths["raw"])
    df = clean(parse_logs(raw))
    df.to_csv(ensure_parent(paths["clean"]), index=False)


if __name__ == "__main__":
    main()
