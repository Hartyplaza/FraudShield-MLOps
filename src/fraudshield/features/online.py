"""
src/fraudshield/features/online.py

Online (one-transaction-at-a-time) feature computation.

The same code path is used for training and serving:
    - training replays historical transactions through FeatureState in time order
    - serving passes each new transaction through the same FeatureState

so the model always sees features computed exactly the same way, using only
information available at the moment the transaction happened.
"""

import json
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from fraudshield.features.engineer import (
    FEATURE_COLS,
    HIGH_AMOUNT_THRESHOLD,
    RISKY_TYPES,
)

CATEGORICAL_COLS = ["txn_type", "location", "device", "log_format", "currency"]
UNKNOWN = "unknown"
WINDOWS = {"1d": timedelta(days=1), "7d": timedelta(days=7), "30d": timedelta(days=30)}


# ---------------------------------------------------------------- encoders
class CategoryEncoder:
    """Fixed category -> integer mapping, learned once from training data.

    Codes are assigned in sorted order (matching pandas .cat.codes), and
    values never seen in training map to -1 instead of shifting other codes.
    """

    def __init__(self, vocab: dict | None = None):
        self.vocab = vocab or {}

    def fit(self, df: pd.DataFrame) -> "CategoryEncoder":
        for col in CATEGORICAL_COLS:
            values = df[col].fillna(UNKNOWN).astype(str)
            self.vocab[col] = sorted(values.unique().tolist())
        return self

    def encode(self, col: str, value) -> int:
        value = UNKNOWN if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
        try:
            return self.vocab[col].index(value)
        except ValueError:
            return -1

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.vocab, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CategoryEncoder":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------- per-user state
@dataclass
class UserState:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0                      # sum of squared deviations (Welford)
    max_amount: float = 0.0
    recent: deque = field(default_factory=deque)   # (timestamp, amount), last 30 days
    last_ts: datetime | None = None
    last_device: str | None = None
    last_location: str | None = None


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


class FeatureState:
    """Holds every user's history and turns one transaction into 27 features."""

    def __init__(self, encoder: CategoryEncoder):
        self.encoder = encoder
        self.users: dict[str, UserState] = {}

    def process(self, txn: dict) -> dict:
        """Compute features for one transaction, then add it to the user's history.

        txn keys: timestamp, user_id, txn_type, amount, currency, location,
                  device, log_format
        """
        ts = pd.Timestamp(txn["timestamp"]).to_pydatetime()
        amount = float(txn["amount"])
        device = None if _missing(txn.get("device")) else txn["device"]
        location = None if _missing(txn.get("location")) else txn["location"]
        s = self.users.setdefault(txn["user_id"], UserState())

        # --- update running amount statistics (inclusive of this transaction)
        s.n += 1
        delta = amount - s.mean
        s.mean += delta / s.n
        s.m2 += delta * (amount - s.mean)
        s.max_amount = max(s.max_amount, amount)
        std = math.sqrt(s.m2 / (s.n - 1)) if s.n > 1 else float("nan")

        # --- velocity: drop transactions older than 30 days, then add this one
        s.recent.append((ts, amount))
        while s.recent and ts - s.recent[0][0] >= WINDOWS["30d"]:
            s.recent.popleft()
        velocity = {}
        for label, width in WINDOWS.items():
            in_window = [a for t, a in s.recent if ts - t < width]
            velocity[f"txn_count_{label}"] = float(len(in_window))
            velocity[f"txn_sum_{label}"] = float(sum(in_window))

        # --- behaviour relative to the previous transaction
        device_changed = int(s.last_device is not None and device != s.last_device)
        location_changed = int(s.last_location is not None and location != s.last_location)
        hrs_since = (ts - s.last_ts).total_seconds() / 3600 if s.last_ts else 0.0

        z = (amount - s.mean) / (std if std and not math.isnan(std) else 1.0)
        if math.isnan(std):
            z = float("nan")

        hour = ts.hour
        feats = {
            "hour": hour,
            "day_of_week": ts.weekday(),
            "is_night": int(0 <= hour <= 5),
            "is_weekend": int(ts.weekday() >= 5),
            "is_business_hours": int(9 <= hour <= 17),
            "amount": amount,
            "amount_z_score": z,
            "is_high_amount": int(amount >= HIGH_AMOUNT_THRESHOLD),
            "is_risky_type": int(txn["txn_type"] in RISKY_TYPES),
            "amount_vs_user_max": amount / (s.max_amount or 1.0),
            "user_mean_amount": s.mean,
            "user_std_amount": std,
            "user_max_amount": s.max_amount,
            **velocity,
            "device_changed": device_changed,
            "location_changed": location_changed,
            "time_since_last_txn_hrs": hrs_since,
            "location_missing": int(location is None),
            "device_missing": int(device is None),
            **{f"{c}_encoded": self.encoder.encode(c, txn.get(c)) for c in CATEGORICAL_COLS},
        }

        # --- remember this transaction for the next one
        s.last_ts = ts
        s.last_device = device
        s.last_location = location
        return {k: feats[k] for k in FEATURE_COLS}


def replay(df: pd.DataFrame, encoder: CategoryEncoder) -> tuple[pd.DataFrame, FeatureState]:
    """Build the training feature matrix by replaying transactions in time order.

    Returns the features (same row order as df) and the final state, which
    holds every user's history up to the last transaction.
    """
    state = FeatureState(encoder)
    order = df.sort_values(["timestamp", "user_id"], kind="stable").index
    rows = {}
    for idx in order:
        rows[idx] = state.process(df.loc[idx].to_dict())
    features = pd.DataFrame.from_dict(rows, orient="index").loc[df.index]
    return features, state