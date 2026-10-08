"""
src/fraudshield/labelling/label_simulator.py

Deterministic fraud label simulation using conjunctive signal rules.

Three rules — each requires multiple signals to fire together:
    Rule A: Night + High amount + Risky transaction type
    Rule B: Night + High amount + Missing location
    Rule C: High amount + Device switch + Missing location

No randomness — identical features always produce identical labels.
This eliminates label noise and gives the model a clean learnable boundary.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Transaction types associated with irreversible money movement
RISKY_TYPES = {"cashout", "withdrawal", "transfer"}

# Amount threshold above which a transaction is considered high-value
HIGH_AMOUNT_THRESHOLD = 4000.0


def _compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute five binary signal flags on the DataFrame.
    Assumes df is sorted by user_id + timestamp.
    """
    # Night: 00:00 to 05:59
    df["sig_night"]          = df["timestamp"].dt.hour.between(0, 5).astype(int)

    # High amount: >= 4,000
    df["sig_high_amount"]    = (df["amount"] >= HIGH_AMOUNT_THRESHOLD).astype(int)

    # Risky transaction type
    df["sig_risky_type"]     = df["txn_type"].isin(RISKY_TYPES).astype(int)

    # Missing location
    df["sig_missing_loc"]    = df["location"].isna().astype(int)

    # Device switch: user changed device from their previous transaction
    df["prev_device"]        = df.groupby("user_id")["device"].shift(1)
    df["sig_device_switch"]  = (
        (df["device"] != df["prev_device"]) &
        df["prev_device"].notna()
    ).astype(int)
    df.drop(columns=["prev_device"], inplace=True)

    return df


def simulate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic fraud rules and add is_fraud column.

    Rule A — Night + High amount + Risky type
        Narrative: large cashout or withdrawal in the early hours

    Rule B — Night + High amount + Missing location
        Narrative: large transaction at night with no traceable origin

    Rule C — High amount + Device switch + Missing location
        Narrative: large transaction on a new device with no location (account takeover)

    Args:
        df: cleaned DataFrame with timestamp, txn_type, amount, location, device

    Returns:
        DataFrame with is_fraud column (0 or 1) added.
    """
    logger.info("Running deterministic label simulation...")

    df = _compute_signals(df)

    rule_a = (
        (df["sig_night"]       == 1) &
        (df["sig_high_amount"] == 1) &
        (df["sig_risky_type"]  == 1)
    )
    rule_b = (
        (df["sig_night"]       == 1) &
        (df["sig_high_amount"] == 1) &
        (df["sig_missing_loc"] == 1)
    )
    rule_c = (
        (df["sig_high_amount"]   == 1) &
        (df["sig_device_switch"] == 1) &
        (df["sig_missing_loc"]   == 1)
    )

    df["is_fraud"] = (rule_a | rule_b | rule_c).astype(int)

    fraud_n = df["is_fraud"].sum()
    legit_n = (df["is_fraud"] == 0).sum()

    logger.info(
        "Labels assigned: fraud=%s (%.1f%%) | legit=%s | ratio=%.0f:1",
        fraud_n, fraud_n / len(df) * 100,
        legit_n, legit_n / fraud_n if fraud_n else float("inf"),
    )

    # Drop intermediate signal columns
    signal_cols = [
        "sig_night", "sig_high_amount", "sig_risky_type",
        "sig_missing_loc", "sig_device_switch",
    ]
    df.drop(columns=signal_cols, inplace=True)

    return df
