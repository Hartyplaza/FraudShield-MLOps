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
import http.client
import json
import logging
import sys
import time
import urllib.parse

import pandas as pd
import yaml

from fraudshield.data.generate import generate
from fraudshield.ingestion.parser import parse_log_line
from fraudshield.labelling.label_simulator import simulate_labels

_connections: dict[str, http.client.HTTPConnection] = {}
ATTEMPTS = 5


def request(method: str, url: str, payload: dict | None = None) -> tuple[int, dict]:
    """One HTTP request over a kept-alive connection (one TLS handshake per run,
    not per request), retried with backoff on network errors.

    A retried POST may have reached the server before the connection broke, so
    a transaction can occasionally be scored twice; for a drill that is noise.
    """
    parts = urllib.parse.urlsplit(url)
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if body else {}
    for attempt in range(ATTEMPTS):
        try:
            conn = _connections.get(parts.netloc)
            if conn is None:
                cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
                # Generous timeout: a free-tier host can take a minute to wake up
                conn = _connections[parts.netloc] = cls(parts.netloc, timeout=90)
            conn.request(method, parts.path or "/", body=body, headers=headers)
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read() or b"{}")
        except (OSError, http.client.HTTPException) as e:
            _connections.pop(parts.netloc, None)
            if attempt == ATTEMPTS - 1:
                raise
            wait = 2 ** attempt
            print(f"  network error ({e}); retry {attempt + 1}/{ATTEMPTS - 1} in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable")


def post(url: str, payload: dict) -> tuple[int, dict]:
    return request("POST", url, payload)


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
    return request("GET", url)


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
    parser.add_argument("--skip", type=int, default=0,
                        help="resume a broken run: skip the first N transactions (already sent)")
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
    latencies, caught, refused, sent_bad = [], [], 0, 0
    i = 0
    for line, valid in interleave(lines, rejected):
        if i < args.skip:            # resuming: these were sent by the broken run
            i += valid
            continue
        t0 = time.perf_counter()
        try:
            status, body = post(f"{args.url}/score", {"log": line})
        except OSError as e:
            raise SystemExit(f"\nNetwork down after {i} transactions ({e}). "
                             f"Resume with the same command plus --skip {i}") from None
        if not valid:
            sent_bad += 1
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
    print(f"Unparseable lines: {sent_bad} sent, {refused} refused with 422")
    print("\nExamples of fraud caught live:")
    for b in caught:
        print(f"  {b['timestamp']}  {b['user_id']}  {b['txn_type']:<10} {b['amount']:>8.2f}  "
              f"p={b['fraud_probability']:.3f}  {b['risk_level']}")
    print_drift(args.url)


if __name__ == "__main__":
    main()