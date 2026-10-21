"""
Stage 1: raw logs -> parsed and cleaned transactions.

    uv run python -m fraudshield.pipeline.ingest
"""

import pandas as pd

from fraudshield.ingestion.cleaner import clean
from fraudshield.ingestion.parser import parse_logs
from fraudshield.pipeline.config import ensure_parent, stage_args
from fraudshield.validation.schemas import DataValidationError, clean_schema, validate


def main() -> None:
    cfg = stage_args("Parse and clean raw transaction logs.")
    paths = cfg["paths"]
    v = cfg["validation"]

    raw = pd.read_csv(paths["raw"])
    df = clean(parse_logs(raw))

    parse_rate = len(df) / len(raw)
    if parse_rate < v["min_parse_rate"]:
        raise DataValidationError(
            f"ingest: only {parse_rate:.1%} of raw rows parsed (minimum {v['min_parse_rate']:.0%})"
        )
    validate(df, clean_schema(v), "ingest")
    df.to_csv(ensure_parent(paths["clean"]), index=False)


if __name__ == "__main__":
    main()
