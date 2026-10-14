"""
Tests for online (one-at-a-time) feature computation.

The key guarantee: features computed by streaming transactions one by one
are identical to the batch features wherever the batch features are valid,
and differ only where the batch code used information from the future.
"""

import logging

import numpy as np
import pandas as pd
import pytest
import yaml

from fraudshield.data.generate import generate
from fraudshield.features.engineer import FEATURE_COLS, build_features
from fraudshield.features.online import CategoryEncoder, FeatureState, replay
from fraudshield.ingestion.cleaner import clean
from fraudshield.ingestion.parser import parse_logs

# Features whose batch definition uses the user's whole history, including
# transactions that happen later (data leakage). Online versions use only the past.
LEAKY = {"user_mean_amount", "user_std_amount", "user_max_amount",
         "amount_z_score", "amount_vs_user_max"}


@pytest.fixture(scope="module")
def transactions():
    logging.disable(logging.INFO)
    with open("configs/pipeline.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["generator"]
    df = clean(parse_logs(pd.DataFrame({"raw_log": generate(cfg)})))
    return df.reset_index(drop=True)


@pytest.fixture(scope="module")
def encoder(transactions):
    return CategoryEncoder().fit(transactions)


@pytest.fixture(scope="module")
def online(transactions, encoder):
    features, _ = replay(transactions, encoder)
    return features


@pytest.fixture(scope="module")
def batch(transactions):
    return build_features(transactions.copy()).reset_index(drop=True)


@pytest.mark.parametrize("col", sorted(set(FEATURE_COLS) - LEAKY))
def test_non_leaky_features_match_batch(col, online, batch):
    assert np.allclose(batch[col].astype(float), online[col].astype(float), equal_nan=True)


@pytest.mark.parametrize("col", ["user_mean_amount", "user_std_amount", "user_max_amount"])
def test_batch_user_stats_use_the_future(col, transactions, online, batch):
    # Batch and online agree only on each user's LAST transaction: that is the
    # only point where "all history" and "history so far" are the same thing.
    last = transactions.sort_values("timestamp").groupby("user_id").tail(1).index
    assert np.allclose(batch.loc[last, col], online.loc[last, col], equal_nan=True)
    assert not np.allclose(batch[col], online[col], equal_nan=True)


def test_user_mean_uses_only_past_and_current(transactions, online):
    user = transactions["user_id"].iloc[0]
    rows = transactions[transactions["user_id"] == user].sort_values("timestamp")
    expected = rows["amount"].expanding().mean().to_numpy()
    assert np.allclose(online.loc[rows.index, "user_mean_amount"], expected)


def test_streaming_in_two_halves_equals_one_replay(transactions, encoder, online):
    ordered = transactions.sort_values(["timestamp", "user_id"], kind="stable")
    state = FeatureState(encoder)
    streamed = {idx: state.process(ordered.loc[idx].to_dict()) for idx in ordered.index}
    streamed = pd.DataFrame.from_dict(streamed, orient="index").loc[transactions.index]
    pd.testing.assert_frame_equal(streamed[FEATURE_COLS], online[FEATURE_COLS])


def test_unseen_category_maps_to_minus_one(encoder):
    assert encoder.encode("device", "Brand New Phone 99") == -1
    assert encoder.encode("txn_type", "withdrawal") == encoder.vocab["txn_type"].index("withdrawal")


def test_encoding_does_not_depend_on_batch(encoder):
    # The bug this replaces: pandas .cat.codes on a one-row batch gives 0 for everything.
    assert encoder.encode("txn_type", "withdrawal") == 7


def test_encoder_round_trip(encoder, tmp_path):
    path = tmp_path / "encoders.json"
    encoder.save(path)
    assert CategoryEncoder.load(path).vocab == encoder.vocab