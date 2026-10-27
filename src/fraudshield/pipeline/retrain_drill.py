"""
Stage 10: retraining drill. Tests the retraining decision as the drift drill
tests the monitor. For each scenario, two labelled weeks of recent traffic
arrive; the rule (retraining/retrain.py) decides to keep the champion,
recalibrate it or retrain it. Then the following two weeks show whether the
decision was right.

The expected decision for each scenario is in configs/pipeline.yaml ->
retraining.drill.expect; any other decision fails the pipeline.

    uv run python -m fraudshield.pipeline.retrain_drill
"""

import copy
import json
import logging

import joblib
import pandas as pd

from fraudshield.evaluation.gates import ModelQualityError
from fraudshield.pipeline.config import (
    ensure_parent,
    normal_traffic,
    now_normal,
    stage_args,
    training_end,
    training_periods,
)
from fraudshield.retraining.retrain import (
    assess,
    build_split,
    decide,
    labelled,
    period,
    scores,
)

logger = logging.getLogger(__name__)


class RetrainDrillError(ModelQualityError):
    """Raised when the retraining rule makes an unexpected decision."""


def future_scores(model, threshold: float, split, future: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Score the next period online, continuing every user's history."""
    state = copy.deepcopy(split.state)
    future = future.sort_values(["timestamp", "user_id"], kind="stable")
    y = labelled(future)["is_fraud"]
    X = pd.DataFrame([state.process(r) for r in future.to_dict("records")])[feature_cols].fillna(-1)
    return scores(model, threshold, X, y.reset_index(drop=True))


def main() -> None:
    cfg = stage_args("Check that the retraining rule makes the right call.")
    paths, rt, gates = cfg["paths"], cfg["retraining"], cfg["gates"]["metrics"]
    drill = rt["drill"]
    limits = {"min_recall": gates["min_recall"], "min_precision": gates["min_precision"],
              "min_holdout_frauds": rt["min_holdout_frauds"],
              "max_base_pr_auc_drop": rt["max_base_pr_auc_drop"]}

    champion = joblib.load(f"{paths['model_dir']}/fraudshield_model.joblib")
    with open(f"{paths['model_dir']}/best_threshold.txt", encoding="utf-8") as f:
        champion_threshold = float(f.read())
    with open(f"{paths['model_dir']}/feature_cols.json", encoding="utf-8") as f:
        feature_cols = json.load(f)
    scenarios = {"none": {}, **cfg["monitoring"]["drill"]["scenarios"]}

    logging.getLogger("fraudshield").setLevel(logging.WARNING)
    base = pd.concat([period(p) for p in training_periods(cfg)], ignore_index=True)
    recent_start, future_start = training_end(cfg), training_end(cfg, plus_days=drill["recent"]["days"])
    report = {}
    normal = normal_traffic(cfg)
    for name, expected in drill["expect"].items():
        change = scenarios[name]
        if now_normal(cfg, change):
            report[name] = {"expected": expected, "decision": "now normal", "skipped": True}
            continue
        recent = period({**normal, **drill["recent"], "start": recent_start, **change})
        future = period({**normal, **drill["future"], "start": future_start, **change})
        split = build_split(base, recent, feature_cols, rt["holdout_days"])
        results = assess(champion, champion_threshold, split, cfg["training"]["min_precision"])
        decision, reason = decide(results, limits)
        chosen = results["champion" if decision == "keep" else decision]
        report[name] = {
            "expected": expected,
            "decision": decision,
            "reason": reason,
            "candidates": {n: {"threshold": r["threshold"], "recent": r["recent"], "base": r["base"]}
                           for n, r in results.items()},
            "future_champion": future_scores(champion, champion_threshold, split, future, feature_cols),
            "future_after_decision": future_scores(chosen["model"], chosen["threshold"], split, future,
                                                   feature_cols),
        }
    logging.getLogger("fraudshield").setLevel(logging.INFO)

    wrong = [n for n, r in report.items() if not r.get("skipped") and r["decision"] != r["expected"]]
    out = {"passed": not wrong, "scenarios": report}
    with open(ensure_parent(paths["retrain_drill"]), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    width = max(map(len, report))
    lines = []
    for n, r in report.items():
        if r.get("skipped"):
            lines.append(f"  {n:<{width}}  now normal (accepted by a retraining), not tested")
            continue
        before, after = r["future_champion"], r["future_after_decision"]
        lines.append(f"  {n:<{width}}  {r['decision']:<12} (expected {r['expected']:<12})  next two weeks: "
                     f"recall {before['recall']:.3f} -> {after['recall']:.3f}, "
                     f"precision {before['precision']:.3f} -> {after['precision']:.3f}")
        lines.append(f"  {'':<{width}}  {r['reason']}")
    summary = "\n".join(lines)
    if wrong:
        logger.error("Retraining drill FAILED (unexpected decision for %s):\n%s", wrong, summary)
        raise RetrainDrillError(f"unexpected retraining decision for {wrong}")
    logger.info("Retraining drill passed:\n%s", summary)


if __name__ == "__main__":
    main()
