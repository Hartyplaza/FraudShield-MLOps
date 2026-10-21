"""
Data contract tests: good data passes, and each kind of bad batch is stopped
with the right check named.
"""

import pandas as pd
import pytest

from fraudshield.data.generate import generate
from fraudshield.features.online import CategoryEncoder, replay
from fraudshield.ingestion.cleaner import clean
from fraudshield.ingestion.parser import parse_logs
from fraudshield.labelling.label_simulator import simulate_labels
from fraudshield.pipeline.config import load_config
from fraudshield.validation.schemas import (
    DataValidationError,
    clean_schema,
    features_schema,
    labelled_schema,
    validate,
)

CFG = load_config()
V = CFG["validation"]


@pytest.fixture(scope="module")
def labelled() -> pd.DataFrame:
    gen = {**CFG["generator"], "n_rows": 3000, "seed": 11}
    raw = pd.DataFrame({"raw_log": generate(gen)})
    return simulate_labels(clean(parse_logs(raw)))


@pytest.fixture(scope="module")
def features(labelled) -> pd.DataFrame:
    feats, _ = replay(labelled, CategoryEncoder().fit(labelled))
    return feats


def failed_checks(df: pd.DataFrame, schema) -> set[str]:
    """Validate a batch that must fail; return 'column: check' for each violation."""
    with pytest.raises(DataValidationError) as err:
        validate(df, schema, "test")
    s = err.value.summary
    return {f"{c}: {k}" for c, k in zip(s["column"], s["check"], strict=True)}


def test_generated_data_meets_every_contract(labelled, features):
    validate(labelled.drop(columns="is_fraud"), clean_schema(V), "ingest")
    validate(labelled, labelled_schema(V), "label")
    validate(features, features_schema(V), "features")


@pytest.mark.parametrize(("corrupt", "expected"), [
    pytest.param(lambda d: d.assign(amount=d["amount"] * 1000), "amount: in_range", id="amount in wrong unit"),
    pytest.param(lambda d: d.assign(amount=-d["amount"]), "amount: in_range", id="negative amount"),
    pytest.param(lambda d: d.assign(txn_type="chargeback"), "txn_type: isin", id="unknown txn_type"),
    pytest.param(lambda d: d.assign(user_id="u-" + d["user_id"]), "user_id: str_matches", id="malformed user_id"),
    pytest.param(lambda d: d.assign(currency="¥"), "currency: isin", id="unknown currency"),
    pytest.param(lambda d: d.assign(log_format="json"), "log_format: isin", id="unknown log format"),
    pytest.param(lambda d: d.drop(columns="amount"), "amount: column_in_dataframe", id="missing column"),
    pytest.param(lambda d: d.head(500), "<batch>: row_count", id="too few rows"),
    pytest.param(lambda d: d.assign(location=None), "<batch>: null_rate(location)", id="location all missing"),
    pytest.param(lambda d: d.assign(is_fraud=0), "<batch>: fraud_rate", id="no fraud at all"),
    pytest.param(lambda d: d.assign(is_fraud=1), "<batch>: fraud_rate", id="everything fraud"),
])
def test_bad_batches_are_stopped(labelled, corrupt, expected):
    failed = failed_checks(corrupt(labelled.copy()), labelled_schema(V))
    assert any(f.startswith(expected) for f in failed), failed


@pytest.mark.parametrize(("corrupt", "expected"), [
    pytest.param(lambda f: f.assign(is_night=2), "is_night: isin", id="binary flag out of range"),
    pytest.param(lambda f: f.assign(txn_count_1d=-1.0), "txn_count_1d: greater_than_or_equal_to", id="negative count"),
    pytest.param(lambda f: f.assign(hour=None), "hour: not_nullable", id="NaN in non-nullable feature"),
    pytest.param(lambda f: f.assign(device_encoded=-5), "device_encoded: greater_than_or_equal_to", id="encoded below -1"),
    pytest.param(lambda f: f.drop(columns="is_risky_type"), "is_risky_type: column_in_dataframe", id="missing feature"),
])
def test_bad_feature_matrices_are_stopped(features, corrupt, expected):
    failed = failed_checks(corrupt(features.copy()), features_schema(V))
    assert any(f.startswith(expected) for f in failed), failed


def test_first_transaction_nans_are_allowed(features):
    # amount_z_score and user_std_amount are undefined for a user's first transaction
    assert features["user_std_amount"].isna().any()
    validate(features, features_schema(V), "features")