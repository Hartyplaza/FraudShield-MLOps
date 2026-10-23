"""
Stage 9: drift drill. Tests the monitor itself, as the quality gate tests the
model:

    1. clean streams (new seeds, nothing changed) must NOT raise an alert
    2. each drift scenario (configs/pipeline.yaml -> monitoring.drill) MUST

Each stream is scored as the API scores it, and the monitor sees the last
`window` transactions, as it does live. Ground-truth labels are available
here, so the report also shows what each drift does to the model's recall:
not every drift harms the model, and the monitor cannot tell which does.

    uv run python -m fraudshield.pipeline.drift_drill
"""

import json
import logging

from fraudshield.evaluation.gates import ModelQualityError
from fraudshield.monitoring.drift import drift_report, load_reference
from fraudshield.monitoring.stream import ScoredStream, score_stream
from fraudshield.pipeline.config import ensure_parent, stage_args
from fraudshield.pipeline.reference import reference_stream_cfg

logger = logging.getLogger(__name__)


class MonitorDrillError(ModelQualityError):
    """Raised when the drift monitor misses a drift or raises a false alarm."""


def run(reference: dict, stream: ScoredStream, mon: dict) -> dict:
    w = mon["window"]
    total = len(stream.features) + stream.unparseable
    # Rejected lines are spread evenly through the stream; count those alongside the window
    unparseable_in_window = round(stream.unparseable * w / len(stream.features))
    report = drift_report(reference, stream.features.tail(w), stream.scores.tail(w), mon,
                          unparseable=unparseable_in_window)
    flagged = stream.scores >= stream.threshold
    fraud = stream.labels == 1
    tp, fp, fn = int((flagged & fraud).sum()), int((flagged & ~fraud).sum()), int((~flagged & fraud).sum())
    return {
        "status": report["status"],
        "alerts": report["alerts"] + [k for k, c in report["checks"].items() if c["status"] == "alert"],
        "max_feature_psi": max(f["psi"] for f in report["features"].values() if f["status"] != "info"),
        "score_psi": report["checks"]["score_psi"]["value"],
        "flag_rate": report["checks"]["flag_rate"]["value"],
        "unparseable_rate": round(stream.unparseable / total, 4),
        "fraud_rate": round(float(fraud.mean()), 4),
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
    }


def main() -> None:
    cfg = stage_args("Check that the drift monitor catches drift and stays quiet otherwise.")
    paths, mon = cfg["paths"], cfg["monitoring"]
    drill = mon["drill"]
    reference = load_reference(paths["reference"])
    base = reference_stream_cfg(cfg)

    logging.getLogger("fraudshield").setLevel(logging.WARNING)
    clean = {s: run(reference, score_stream(paths["model_dir"], {**base, "seed": s}), mon)
             for s in drill["clean_seeds"]}
    scenarios = {name: run(reference, score_stream(paths["model_dir"],
                                                   {**base, "seed": drill["seed"], **change}), mon)
                 for name, change in drill["scenarios"].items()}
    logging.getLogger("fraudshield").setLevel(logging.INFO)

    false_alarms = [s for s, r in clean.items() if r["status"] == "alert"]
    missed = [n for n, r in scenarios.items() if r["status"] != "alert"]
    report = {
        "passed": not false_alarms and not missed,
        "clean_streams": len(clean),
        "false_alarms": len(false_alarms),
        "clean_max_feature_psi": max(r["max_feature_psi"] for r in clean.values()),
        "clean_recall_mean": round(sum(r["recall"] for r in clean.values()) / len(clean), 4),
        "scenarios_detected": len(scenarios) - len(missed),
        "scenarios": scenarios,
    }
    with open(ensure_parent(paths["drift_drill"]), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    width = max(map(len, scenarios))
    clean_label = f"clean ({len(clean)} seeds)"
    lines = [(f"  {clean_label:<{width}}  false alarms {len(false_alarms)}, "
              f"max PSI {report['clean_max_feature_psi']:.3f}, recall {report['clean_recall_mean']:.3f}")]
    lines += [(f"  {n:<{width}}  {r['status']:<5}  recall {r['recall']:.3f}  flag rate {r['flag_rate']:.3f}  "
               f"alerts: {', '.join(r['alerts'][:4])}") for n, r in scenarios.items()]
    summary = "\n".join(lines)
    if not report["passed"]:
        logger.error("Drift drill FAILED (false alarms: %s; missed: %s):\n%s",
                     false_alarms, missed, summary)
        raise MonitorDrillError(f"{len(false_alarms)} false alarm(s), {len(missed)} missed drift(s)")
    logger.info("Drift drill passed:\n%s", summary)


if __name__ == "__main__":
    main()
