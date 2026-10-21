"""
src/fraudshield/evaluation/gates.py

Quality gates a newly trained model must pass before it is accepted:

    1. metric gates: test-set metrics within the limits in configs/pipeline.yaml
    2. golden cases: a fixed set of transactions whose correct decision is not in
       doubt (configs/golden_cases.yaml), scored end to end from raw log lines

A model can have a good average score and still get an obvious case wrong; the
golden cases catch that.
"""

import logging

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