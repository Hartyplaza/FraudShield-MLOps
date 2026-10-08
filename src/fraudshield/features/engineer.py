"""
src/fraudshield/features/engineer.py

Feature engineering pipeline for fraud detection.
Builds 28 features across 6 groups from the cleaned, labelled DataFrame.

Feature groups:
    1. Temporal          - hour, day_of_week, is_night, is_weekend, is_business_hours
    2. Amount            - z-score, high_amount flag, ratio to user max
    3. Velocity          - rolling txn count and sum over 1d, 7d, 30d windows
    4. Behavioural       - device switch, location switch, time since last txn
    5. Missing flags     - location_missing, device_missing
    6. Categorical codes - txn_type, location, device, log_format, currency
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

HIGH_AMOUNT_THRESHOLD = 4000.0
VELOCITY_WINDOWS      = ["1D", "7D", "30D"]

# Exact 28 feature columns the model expects at inference time
FEATURE_COLS = [
    # Temporal
    "hour", "day_of_week", "is_night", "is_weekend", "is_business_hours",
    # Amount
    "amount", "amount_z_score", "is_high_amount", "amount_vs_user_max",
    "user_mean_amount", "user_std_amount", "user_max_amount",
    # Velocity
    "txn_count_1d", "txn_sum_1d", "txn_count_7d", "txn_sum_7d",
    "txn_count_30d", "txn_sum_30d",
    # Behavioural
    "device_changed", "location_changed", "time_since_last_txn_hrs",
    # Missing flags
    "location_missing", "device_missing",
    # Encoded categoricals
    "txn_type_encoded", "location_encoded", "device_encoded",
    "log_format_encoded", "currency_encoded",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from the timestamp column.

    - hour: raw hour (0-23). Fraud spikes between 00:00-05:59.
    - day_of_week: 0=Monday, 6=Sunday.
    - is_night: 1 if hour is 00:00-05:59.
    - is_weekend: 1 if Saturday or Sunday.
    - is_business_hours: 1 if hour is 09:00-17:00.
    """
    logger.info("Adding temporal features...")
    df["hour"]              = df["timestamp"].dt.hour
    df["day_of_week"]       = df["timestamp"].dt.dayofweek
    df["is_night"]          = df["hour"].between(0, 5).astype(int)
    df["is_weekend"]        = (df["day_of_week"] >= 5).astype(int)
    df["is_business_hours"] = df["hour"].between(9, 17).astype(int)
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Contextualise amount relative to each user's own history.

    A 4,000 transaction may be normal for a high-spending user but
    highly suspicious for someone who usually spends 200. We capture
    both the absolute value and the per-user deviation.

    - user_mean_amount: user's average transaction amount
    - user_std_amount: standard deviation of user's amounts
    - user_max_amount: user's historical maximum
    - amount_z_score: standard deviations above/below user's mean
    - is_high_amount: absolute flag for amounts >= 4,000
    - amount_vs_user_max: current amount as fraction of user's max
    """
    logger.info("Adding amount features...")

    user_stats = df.groupby("user_id")["amount"].agg(
        user_mean_amount="mean",
        user_std_amount="std",
        user_max_amount="max",
    ).reset_index()

    df = df.merge(user_stats, on="user_id", how="left")

    # Replace zero std with 1 to avoid division error for single-transaction users
    df["amount_z_score"]     = (
        (df["amount"] - df["user_mean_amount"]) /
        df["user_std_amount"].replace(0, 1)
    )
    df["is_high_amount"]     = (df["amount"] >= HIGH_AMOUNT_THRESHOLD).astype(int)
    df["amount_vs_user_max"] = df["amount"] / df["user_max_amount"].replace(0, 1)

    return df


def add_velocity_features(df: pd.DataFrame,
                           windows: list = None) -> pd.DataFrame:
    """
    Per-user rolling transaction count and sum over time windows.

    Velocity measures how many transactions and how much money a user
    moved in a recent window. Burst activity is a classic fraud pattern.

    Uses time-based rolling windows (requires timestamp as index).
    min_periods=1 ensures first transactions still get a count.

    Args:
        df: DataFrame sorted by user_id + timestamp
        windows: list of pandas offset strings e.g. ['1D', '7D', '30D']
    """
    if windows is None:
        windows = VELOCITY_WINDOWS

    logger.info("Adding velocity features for windows: %s...", windows)

    # Set timestamp as index for time-based rolling
    df = df.set_index("timestamp")

    for window in windows:
        label = window.lower()
        df["txn_count_{}".format(label)] = (
            df.groupby("user_id")["amount"]
              .transform(lambda x: x.rolling(window, min_periods=1).count())
        )
        df["txn_sum_{}".format(label)] = (
            df.groupby("user_id")["amount"]
              .transform(lambda x: x.rolling(window, min_periods=1).sum())
        )

    df = df.reset_index()
    return df


def add_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Capture sudden changes in a user's behaviour.

    These are the strongest indicators of account takeover fraud:
    - device_changed: user switched device from previous transaction
    - location_changed: user switched city from previous transaction
    - time_since_last_txn_hrs: hours since this user's previous transaction
      (very short gaps signal automated fraud; long gaps signal dormant accounts)

    Uses shift(1) within each user group to access the previous row.
    First transaction per user gets 0 (no previous transaction to compare).
    """
    logger.info("Adding behavioural features...")

    df["prev_device"]    = df.groupby("user_id")["device"].shift(1)
    df["prev_location"]  = df.groupby("user_id")["location"].shift(1)
    df["prev_timestamp"] = df.groupby("user_id")["timestamp"].shift(1)

    df["device_changed"] = (
        (df["device"] != df["prev_device"]) &
        df["prev_device"].notna()
    ).astype(int)

    df["location_changed"] = (
        (df["location"] != df["prev_location"]) &
        df["prev_location"].notna()
    ).astype(int)

    df["time_since_last_txn_hrs"] = (
        (df["timestamp"] - df["prev_timestamp"])
        .dt.total_seconds() / 3600
    )

    # Fill NaN for first transaction per user
    df["device_changed"] = df["device_changed"].fillna(0)
    df["location_changed"] = df["location_changed"].fillna(0)
    df["time_since_last_txn_hrs"] = df["time_since_last_txn_hrs"].fillna(0)

    df.drop(columns=["prev_device", "prev_location", "prev_timestamp"],
            inplace=True)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical columns to integer codes for XGBoost/LightGBM.

    Uses pandas category dtype with .cat.codes — stable integer assignment
    per unique value. Missing values are filled with 'unknown' first so they
    get their own code rather than -1.

    Why not one-hot encoding? Tree-based models handle ordinal integer codes
    natively and split on them efficiently. One-hot adds 20+ columns for
    negligible gain with gradient boosting.
    """
    logger.info("Encoding categorical features...")

    # Fill NaN with 'unknown' before encoding — preserves missingness info
    df["location"] = df["location"].fillna("unknown")
    df["device"]   = df["device"].fillna("unknown")
    df["currency"] = df["currency"].fillna("unknown")

    for col in ["txn_type", "location", "device", "log_format", "currency"]:
        df[col + "_encoded"] = df[col].astype("category").cat.codes

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Args:
        df: cleaned, labelled DataFrame sorted by user_id + timestamp

    Returns:
        DataFrame with all 28 engineered features added.
        Original columns are preserved alongside new feature columns.
    """
    logger.info("Building feature matrix...")
    df = add_temporal_features(df)
    df = add_amount_features(df)
    df = add_velocity_features(df)
    df = add_behavioural_features(df)
    df = encode_categoricals(df)
    logger.info("Feature matrix shape: %s", df.shape)
    return df
