"""
src/fraudshield/serving/stream_demo.py

Live demo: stream new, unseen transactions into the running API one at a time
and watch it separate fraud from legitimate traffic.

The stream is a fresh period of synthetic logs that starts after the training
data ends, from a different random seed. Ground truth comes from the same
labelling rules used in training, applied to the stream (a user's first
transaction in the stream does not see their pre-stream device, so a few
labels can differ from what the full history would give).

Usage (with the API running on port 8000):
    uv run python -m fraudshield.serving.stream_demo

Drift drill against any deployment: stream a clean period, then a drifted one
(a scenario from configs/pipeline.yaml -> monitoring.drill), and read the
API's drift report after each:
    uv run python -m fraudshield.serving.stream_demo --url <URL>
    uv run python -m fraudshield.serving.stream_demo --url <URL> --scenario amounts_x3 --start 2025-08-14
"""

import argparse
import json
import logging
import time
import urllib.error
import urllib.request

import pandas as pd
import yaml

from fraudshield.data.generate import generate
from fraudshield.ingestion.parser import parse_log_line
from fraudshield.labelling.label_simulator import simulate_labels


def post(url: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        # Generous timeout: a free-tier host can take a minute to wake up
        with urllib.request.urlopen(req, timeout=90) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def build_stream(cfg: dict) -> tuple[list[str], list[str], pd.Series]:
    """Return (valid lines in time order, rejected lines, ground-truth labels)."""
    lines = [r for r in generate(cfg) if r is not None]
    parsed = [(line, parse_log_line(line)) for line in lines]
    valid = sorted(((ln, p) for ln, p in parsed if p), key=lambda r: (r[1]["timestamp"], r[1]["user_id"]))
    rejected = [ln for ln, p in parsed if p is None]

    df = pd.DataFrame([p for _, p in valid])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    ordered = df.sort_values(["user_id", "timestamp"], kind="stable")
    labels = simulate_labels(ordered.copy())["is_fraud"].reindex(df.index)
    return [ln for ln, _ in valid], rejected, labels


def get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=90) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def interleave(lines: list[str], rejected: list[str]) -> list[tuple[str, bool]]:
    """Spread the unparseable lines evenly through the valid ones, as in real traffic."""
    out, step = [], (len(lines) / len(rejected)) if rejected else 0
    j = 0
    for i, line in enumerate(lines):
        while j < len(rejected) and j * step <= i:
            out.append((rejected[j], False))
            j += 1
        out.append((line, True))
    return out + [(r, False) for r in rejected[j:]]


def print_drift(url: str) -> None:
    status, report = get(f"{url}/monitoring/drift")
    if status != 200:
        print(f"\nDrift report unavailable ({status})")
        return
    print(f"\nDrift report: {report['status'].upper()} over the last {report['n']} scored transactions")
    if "checks" in report:
        for name, c in report["checks"].items():
            print(f"  {name:<17} {c['value']:.4f} (reference {c.get('reference', '-')})  {c['status']}")
        print(f"  alerts: {', '.join(report['alerts']) or 'none'}")
        print(f"  warnings: {', '.join(report['warnings']) or 'none'}")
    else:
        print(f"  {report.get('detail', '')}")


def tally(tp: int, fp: int, fn: int) -> str:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return f"caught {tp:3d} | missed {fn:3d} | false alarms {fp:3d} | precision {precision:.2f} | recall {recall:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream unseen transactions into the API.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start", default="2025-07-31")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--n-rows", type=int, default=2500)
    parser.add_argument("--every", type=int, default=250, help="print progress every N transactions")
    parser.add_argument("--scenario", help="apply a drift scenario from monitoring.drill.scenarios")
    args = parser.parse_args()
    logging.disable(logging.INFO)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    cfg = {**config["generator"], "seed": args.seed, "start": args.start,
           "days": args.days, "n_rows": args.n_rows}
    if args.scenario:
        cfg.update(config["monitoring"]["drill"]["scenarios"][args.scenario])

    lines, rejected, labels = build_stream(cfg)
    label = f", scenario {args.scenario}" if args.scenario else ""
    print(f"Streaming {len(lines)} transactions from {args.start} ({args.days} days, seed {args.seed}{label})")
    print(f"Actual fraud in stream: {int(labels.sum())}; unparseable lines mixed in: {len(rejected)}\n")

    tp = fp = fn = 0
    latencies, caught, refused = [], [], 0
    i = 0
    for line, valid in interleave(lines, rejected):
        t0 = time.perf_counter()
        status, body = post(f"{args.url}/score", {"log": line})
        if not valid:
            refused += status == 422
            continue
        i += 1
        latencies.append((time.perf_counter() - t0) * 1000)
        if status != 200:
            raise SystemExit(f"API returned {status}: {body}")

        actual = bool(labels.iloc[i - 1])
        if body["is_fraud"] and actual:
            tp += 1
            if len(caught) < 5:
                caught.append(body)
        elif body["is_fraud"]:
            fp += 1
        elif actual:
            fn += 1

        if i % args.every == 0 or i == len(lines):
            print(f"[{i:5d}/{len(lines)}] {tally(tp, fp, fn)}")

    lat = pd.Series(latencies)
    print(f"\nLatency per request: p50 {lat.quantile(0.5):.1f} ms | p95 {lat.quantile(0.95):.1f} ms")
    print(f"Unparseable lines: {len(rejected)} sent, {refused} refused with 422")
    print("\nExamples of fraud caught live:")
    for b in caught:
        print(f"  {b['timestamp']}  {b['user_id']}  {b['txn_type']:<10} {b['amount']:>8.2f}  "
              f"p={b['fraud_probability']:.3f}  {b['risk_level']}")
    print_drift(args.url)


if __name__ == "__main__":
    main()