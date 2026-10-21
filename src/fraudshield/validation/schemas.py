"""
src/fraudshield/validation/schemas.py

Data contracts checked between pipeline stages. A batch that breaks the contract
stops the pipeline with a readable report, instead of quietly training a worse
model.

Two kinds of check:
    - row checks: types, allowed values, ranges (every row must pass)
    - batch checks: row count, null rates, fraud rate (the batch as a whole)

Thresholds and allowed values live in configs/pipeline.yaml -> validation, so
changing the contract is a reviewed config change, tracked by DVC.
"""

import logging

import pandas as pd
import pandera.pandas as pa

from fraudshield.features.engineer import FEATURE_COLS

logger = logging.getLogger(__name__)

BINARY_FEATURES = [
    "is_night", "is_weekend", "is_business_hours", "is_high_amount", "is_risky_type",
    "device_changed", "location_changed", "location_missing", "device_missing",
]
ENCODED_FEATURES = [c for c in FEATURE_COLS if c.endswith("_encoded")]
# Undefined for a user's first transaction (no spread yet), so NaN is expected there
NULLABLE_FEATURES = ["amount_z_score", "user_std_amount"]


class DataValidationError(ValueError):
    """Raised when a batch breaks the data contract.

    .summary holds one row per violated check: column, check, failures, examples.
    """

    def __init__(self, message: str, summary: pd.DataFrame | None = None):
        super().__init__(message)
        self.summary = summary


def _null_rate_checks(max_null_rate: dict) -> list[pa.Check]:
    return [
        pa.Check(lambda df, col=col, limit=limit: df[col].isna().mean() <= limit,
                 name=f"null_rate({col}) <= {limit}")
        for col, limit in max_null_rate.items()
    ]


def _clean_columns(v: dict) -> dict:
    lo, hi = v["amount_range"]
    return {
        "timestamp": pa.Column("datetime64[us]"),
        "user_id": pa.Column(str, pa.Check.str_matches(r"^user\d+$")),
        "txn_type": pa.Column(str, pa.Check.isin(v["txn_types"])),
        "amount": pa.Column(float, pa.Check.in_range(lo, hi)),
        "currency": pa.Column(str, pa.Check.isin(v["currencies"]), nullable=True),
        "location": pa.Column(str, nullable=True),
        "device": pa.Column(str, nullable=True),
        "log_format": pa.Column(str, pa.Check.isin(v["log_formats"])),
        "location_missing": pa.Column(int, pa.Check.isin([0, 1])),
        "device_missing": pa.Column(int, pa.Check.isin([0, 1])),
    }


def _clean_checks(v: dict) -> list[pa.Check]:
    return [
        pa.Check(lambda df: len(df) >= v["min_rows"], name=f"row_count >= {v['min_rows']}"),
        *_null_rate_checks(v["max_null_rate"]),
    ]


def clean_schema(v: dict) -> pa.DataFrameSchema:
    """Contract for parsed and cleaned transactions (output of ingest)."""
    return pa.DataFrameSchema(_clean_columns(v), checks=_clean_checks(v), strict=False)


def labelled_schema(v: dict) -> pa.DataFrameSchema:
    """Contract for labelled transactions: clean contract plus a sane fraud rate."""
    lo, hi = v["fraud_rate"]
    columns = {**_clean_columns(v), "is_fraud": pa.Column(int, pa.Check.isin([0, 1]))}
    fraud_rate = pa.Check(lambda df: lo <= df["is_fraud"].mean() <= hi,
                          name=f"fraud_rate in [{lo}, {hi}]")
    return pa.DataFrameSchema(columns, checks=[*_clean_checks(v), fraud_rate], strict=False)


def features_schema(v: dict) -> pa.DataFrameSchema:
    """Contract for the feature matrix the model trains on."""
    columns = {}
    for col in FEATURE_COLS:
        checks = []
        if col in BINARY_FEATURES:
            checks.append(pa.Check.isin([0, 1]))
        elif col in ENCODED_FEATURES:
            checks.append(pa.Check.ge(-1))
        elif col.startswith(("txn_count", "txn_sum", "time_since")):
            checks.append(pa.Check.ge(0))
        columns[col] = pa.Column(float, checks, nullable=col in NULLABLE_FEATURES, coerce=True)
    return pa.DataFrameSchema(columns, strict=False)


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema, stage: str) -> pd.DataFrame:
    """Validate a batch; on failure log a summary and raise DataValidationError."""
    try:
        out = schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        cases = err.failure_cases
        # A missing column is reported at batch level; name the column instead
        missing = cases["check"] == "column_in_dataframe"
        column = cases["column"].where(~missing, cases["failure_case"]).fillna("<batch>")
        cases = cases.assign(column=column)
        summary = (cases.groupby(["column", "check"])
                   .agg(failures=("failure_case", "size"),
                        examples=("failure_case", lambda s: list(s.dropna().unique()[:3])))
                   .reset_index())
        logger.error("Data validation failed at stage '%s':\n%s", stage, summary.to_string(index=False))
        raise DataValidationError(
            f"{stage}: {len(summary)} contract violation(s), "
            f"{int(summary['failures'].sum())} failing value(s)",
            summary,
        ) from None
    logger.info("Data validation passed at stage '%s' (%d rows)", stage, len(df))
    return out