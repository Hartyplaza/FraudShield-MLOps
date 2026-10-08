"""
src/fraudshield/ingestion/cleaner.py

Post-parse cleaning: type normalisation, sorting, and missing value flags.
Input:  raw parsed DataFrame from parser.parse_logs()
Output: clean DataFrame ready for feature engineering
"""

import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Fields that must be present and non-null for a row to be usable
CRITICAL_COLS = ["timestamp", "user_id", "txn_type", "amount"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning pipeline on a parsed log DataFrame.

    Steps:
        1. Drop rows missing any critical field
        2. Normalise txn_type to lowercase
        3. Sort by user_id + timestamp (required for sequential features)
        4. Add binary missing-value flags for location and device

    Args:
        df: DataFrame produced by parser.parse_logs()

    Returns:
        Cleaned DataFrame with location_missing and device_missing columns added.
    """
    logger.info("Running cleaning pipeline on %s rows...", f"{len(df):,}")

    # Step 1: drop rows missing critical fields
    before = len(df)
    df = df.dropna(subset=CRITICAL_COLS)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %s rows with missing critical fields", dropped)

    # Step 2: normalise txn_type casing
    df["txn_type"] = df["txn_type"].str.lower().str.strip()

    # Step 3: sort by user + time — mandatory for all sequential operations
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Step 4: flag missing optional fields as explicit binary features
    # Missingness is informative — legitimate apps almost always report
    # location and device. A missing value is itself a fraud signal.
    df["location_missing"] = df["location"].isna().astype(int)
    df["device_missing"]   = df["device"].isna().astype(int)

    logger.info(
        "Clean dataset: %s rows | %s columns | "
        "location_missing=%s | device_missing=%s",
        f"{len(df):,}",
        df.shape[1],
        df["location_missing"].sum(),
        df["device_missing"].sum(),
    )
    return df
