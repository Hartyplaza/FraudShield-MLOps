"""
src/fraudshield/evaluation/gates.py

Quality gates a newly trained model must pass before it is accepted:

    1. metric gates: test-set metrics within the limits in configs/pipeline.yaml
    2. golden cases: a fixed set of transactions whose correct decision is not in
       doubt (configs/golden_cases.yaml), scored end to end from raw log lines
    3. invariance: the same transaction written in any log format or currency
       gets the same score (configs/behaviour_cases.yaml)
    4. directional: making a transaction riskier never lowers its score

A model can have a good average score and still get an obvious case wrong, or
lean on something irrelevant; checks 2-4 catch that.
"""

import logging
from datetime import datetime
from itertools import count

from fraudshield.data.formats import FORMATS, NO_CURRENCY, render

CURRENCIES = ("\u00a3", "\u20ac", "$")

logger = logging.getLogger(__name__)


class ModelQualityError(RuntimeError):
    """Raised when a trained model fails a quality gate."""


def check_metrics(metrics: dict, gates: dict) -> list[dict]:
    """Compare test metrics with the limits. Keys: min_<metric> or max_<metric>."""
    results = []
    for key, limit in gates.items():
        kind, metric = key.split("_", 1)
        value = metrics[metric]
        passed = value >= limit if kind == "min" else value <= limit
        results.append({"check": f"{metric} {'>=' if kind == 'min' else '<='} {limit}",
                        "value": value, "passed": passed})
    return results


def check_golden(scorer, cases: list[dict]) -> list[dict]:
    """Score each golden case through `scorer` (a RawScorer) and compare with the expected decision."""
    results = []
    for case in cases:
        for line in case.get("history", []):
            scorer.score(line)
        out = scorer.score(case["line"])
        expected_fraud = case["expect"] == "fraud"
        results.append({"check": f"golden: {case['name']} -> {case['expect']}",
                        "value": out["fraud_probability"],
                        "passed": out["is_fraud"] == expected_fraud})
    return results


def summarise(results: list[dict]) -> str:
    width = max(len(r["check"]) for r in results)
    return "\n".join(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['check']:<{width}}  {r['value']:.4f}"
                     for r in results)


# ---------------------------------------------------------------- behaviour checks
def _score(scorer, history: list[dict], txn: dict, user: str, fmt: str, currency: str) -> float:
    for h in history:
        scorer.score(render({**h, "user_id": user, "currency": currency}, fmt))
    return scorer.score(render({**txn, "user_id": user, "currency": currency}, fmt))["fraud_probability"]


def _parse_case(case: dict) -> tuple[list[dict], dict]:
    def rec(d: dict) -> dict:
        return {**d, "timestamp": datetime.fromisoformat(d["timestamp"])}
    return [rec(h) for h in case.get("history", [])], rec(case["txn"])


def check_invariance(scorer, cases: list[dict], tolerance: float) -> list[dict]:
    """The same transaction written in every log format and currency must get the
    same score (within `tolerance`): how a log line is written is not evidence."""
    users = (f"user{n}" for n in count(800000))
    results = []
    for case in cases:
        history, txn = _parse_case(case)
        scores = [_score(scorer, history, txn, next(users), fmt, cur)
                  for fmt in FORMATS for cur in CURRENCIES
                  if not (fmt in NO_CURRENCY and cur != CURRENCIES[0])]
        spread = max(scores) - min(scores)
        results.append({"check": f"invariance: {case['name']} (format, currency) spread <= {tolerance}",
                        "value": spread, "passed": spread <= tolerance})
    return results


def check_directional(scorer, cases: list[dict]) -> list[dict]:
    """Changing one field in a riskier direction must not lower the score."""
    users = (f"user{n}" for n in count(900000))
    results = []
    for case in cases:
        history, txn = _parse_case(case)
        _, riskier = _parse_case({"txn": {**case["txn"], **case["riskier"]}})
        fmt, cur = FORMATS[2], CURRENCIES[0]
        before = _score(scorer, history, txn, next(users), fmt, cur)
        after = _score(scorer, history, riskier, next(users), fmt, cur)
        results.append({"check": f"directional: {case['name']} (score must not drop)",
                        "value": after - before, "passed": after >= before})
    return results