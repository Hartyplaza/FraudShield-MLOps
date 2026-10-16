"""
src/fraudshield/evaluation/stream_errors.py

Error analysis for the live stream: which frauds does the model miss, which
labelling rule produced them, and what did the model see at the time?

Replays the same stream as stream_demo.py through RawScorer in-process (no API
needed). Streaming gives the same features as batch, so the results match the
live demo.

Usage (after `make all`):
    uv run python -m fraudshield.evaluation.stream_errors
"""

import argparse
import logging

import pandas as pd
import yaml

from fraudshield.ingestion.parser import parse_log_line
from fraudshield.labelling.label_simulator import HIGH_AMOUNT_THRESHOLD, RISKY_TYPES
from fraudshield.serving.predictor import FraudPredictor
from fraudshield.serving.scorer import RawScorer
from fraudshield.serving.stream_demo import build_stream

SHOW = ["hour", "amount", "txn_type", "location_missing", "device_changed",
        "amount_z_score", "amount_vs_user_max", "user_mean_amount"]


def rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the three labelling rules on the stream, in stream time order."""
    d = df.sort_values(["user_id", "timestamp"], kind="stable").copy()
    night = d["timestamp"].dt.hour.between(0, 5)
    high = d["amount"] >= HIGH_AMOUNT_THRESHOLD
    risky = d["txn_type"].isin(RISKY_TYPES)
    no_loc = d["location"].isna()
    prev = d.groupby("user_id")["device"].shift(1)
    switch = (d["device"] != prev) & prev.notna()
    return pd.DataFrame({
        "rule_a": night & high & risky,
        "rule_b": night & high & no_loc,
        "rule_c": high & switch & no_loc,
        "label_switch": switch,
    }).reindex(df.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse missed frauds in the stream.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start", default="2025-07-31")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--n-rows", type=int, default=2500)
    args = parser.parse_args()
    logging.disable(logging.INFO)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["generator"]
    cfg.update(seed=args.seed, start=args.start, days=args.days, n_rows=args.n_rows)

    lines, _, labels = build_stream(cfg)
    predictor = FraudPredictor(args.model_dir)
    predictor.load()
    scorer = RawScorer.from_model_dir(predictor, args.model_dir)

    rows = []
    for line in lines:
        # Same steps as RawScorer.score, but keeping the features for analysis.
        parsed = parse_log_line(line)
        feats = scorer.state.process(parsed)
        prob = predictor.predict(pd.DataFrame([feats]))["fraud_probability"][0]
        rows.append({**parsed, **feats, "prob": prob})

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["actual"] = labels.to_numpy().astype(bool)
    df["pred"] = df["prob"] >= predictor.threshold
    df = df.join(rule_flags(df))
    df["rules"] = df[["rule_a", "rule_b", "rule_c"]].apply(
        lambda r: "+".join(k[-1].upper() for k, v in r.items() if v) or "-", axis=1)

    fraud = df[df["actual"]]
    print(f"Threshold {predictor.threshold} | fraud {len(fraud)} | caught {int(fraud['pred'].sum())}\n")
    print("Recall by rule combination:")
    by = fraud.groupby("rules")["pred"].agg(total="size", caught="sum")
    by["missed"] = by["total"] - by["caught"]
    print(by.to_string(), "\n")

    missed = fraud[~fraud["pred"]].sort_values("prob", ascending=False)
    print(f"Missed frauds ({len(missed)}):")
    print(missed[["user_id", "rules", "prob", *SHOW]].round(3).to_string(index=False), "\n")

    # Rule C depends on a device switch. The label sees only stream history;
    # the model's state also holds pre-stream history. Count disagreements.
    c = fraud[fraud["rule_c"]]
    disagree = c[c["device_changed"] != c["label_switch"].astype(int)]
    print(f"Rule C frauds where label and model disagree on device switch: {len(disagree)}")

    fp = df[df["pred"] & ~df["actual"]]
    print(f"\nFalse alarms ({len(fp)}): how many of the three rules' signals each has")
    night = fp["timestamp"].dt.hour.between(0, 5)
    high = fp["amount"] >= HIGH_AMOUNT_THRESHOLD
    risky = fp["txn_type"].isin(RISKY_TYPES)
    no_loc = fp["location_missing"] == 1
    sw = fp["label_switch"]
    near = pd.DataFrame({
        "A(night,high,risky)": night.astype(int) + high + risky,
        "B(night,high,noloc)": night.astype(int) + high + no_loc,
        "C(high,switch,noloc)": high.astype(int) + sw + no_loc,
    })
    print(near.max(axis=1).value_counts().rename("max signals of 3").to_string())


if __name__ == "__main__":
    main()