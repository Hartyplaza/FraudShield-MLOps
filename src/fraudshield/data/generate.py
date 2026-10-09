"""
src/fraudshield/data/generate.py

Generate synthetic raw transaction logs that match the parser's 7 formats.

The 'assessment' profile is a statistical twin of the original assessment
file: every field is drawn independently, so the deterministic labelling
rules produce the same fraud prevalence as the original data.

Usage:
    uv run python -m fraudshield.data.generate --config configs/pipeline.yaml --out data/raw/logs.csv
"""

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

from fraudshield.data.formats import FORMATS, NO_CURRENCY, render

MALFORMED = "MALFORMED_LOG"


def _pick(rng: np.random.Generator, options: list, missing_rate: float = 0.0):
    """Choose one option uniformly, or None with probability missing_rate."""
    if missing_rate and rng.random() < missing_rate:
        return None
    return options[rng.integers(len(options))]


def make_record(rng: np.random.Generator, cfg: dict, user_ids: list,
                user_weights: np.ndarray, fmt: str) -> dict:
    """Draw one transaction record. Every field is independent of the others."""
    start = datetime.fromisoformat(cfg["start"])
    offset_s = int(rng.integers(cfg["days"] * 24 * 3600))

    return {
        "timestamp": start + timedelta(seconds=offset_s),
        "user_id":   user_ids[rng.choice(len(user_ids), p=user_weights)],
        "txn_type":  _pick(rng, cfg["txn_types"]),
        "amount":    round(float(rng.uniform(cfg["amount_min"], cfg["amount_max"])), 2),
        "currency":  None if fmt in NO_CURRENCY else _pick(rng, cfg["currencies"]),
        "location":  _pick(rng, cfg["locations"], cfg["location_missing_rate"]),
        "device":    _pick(rng, cfg["devices"], cfg["device_missing_rate"]),
    }


def generate(cfg: dict) -> list:
    """Return a list of raw log strings (None for null rows)."""
    rng = np.random.default_rng(cfg["seed"])

    user_ids = [f"user{1000 + i}" for i in range(cfg["n_users"])]
    # Uneven activity: some users transact far more than others
    activity = rng.gamma(shape=3.0, scale=1.0, size=cfg["n_users"])
    user_weights = activity / activity.sum()

    rows = []
    for _ in range(cfg["n_rows"]):
        u = rng.random()
        if u < cfg["null_rate"]:
            rows.append(None)
        elif u < cfg["null_rate"] + cfg["malformed_rate"]:
            rows.append(MALFORMED)
        else:
            fmt = FORMATS[rng.integers(len(FORMATS))]
            record = make_record(rng, cfg, user_ids, user_weights, fmt)
            rows.append(render(record, fmt))
    return rows


def write_csv(rows: list, out: Path) -> None:
    """Write rows to a single-column CSV named raw_log. None becomes an empty cell."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["raw_log"])
        for row in rows:
            writer.writerow(["" if row is None else row])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic raw transaction logs.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--out", default="data/raw/logs.csv")
    parser.add_argument("--seed", type=int, help="override generator.seed")
    parser.add_argument("--n-rows", type=int, help="override generator.n_rows")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["generator"]
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.n_rows is not None:
        cfg["n_rows"] = args.n_rows

    rows = generate(cfg)
    write_csv(rows, Path(args.out))
    print(f"Wrote {len(rows):,} rows to {args.out} (seed={cfg['seed']})")


if __name__ == "__main__":
    main()