"""
Stage 5: quality gate. Checks the trained model against the metric limits, the
golden cases and the behaviour checks; fails the pipeline if any check fails.

Runs as its own stage, so changing a limit or a case re-checks the existing
model without retraining it.

    uv run python -m fraudshield.pipeline.gate
"""

import json
import logging

import yaml

from fraudshield.evaluation.gates import (
    ModelQualityError,
    check_directional,
    check_golden,
    check_invariance,
    check_metrics,
    summarise,
)
from fraudshield.pipeline.config import ensure_parent, stage_args
from fraudshield.serving.predictor import FraudPredictor
from fraudshield.serving.scorer import RawScorer

logger = logging.getLogger(__name__)


def main() -> None:
    cfg = stage_args("Check the trained model against the quality gates.")
    paths, gates = cfg["paths"], cfg["gates"]

    with open(paths["metrics"], encoding="utf-8") as f:
        metrics = json.load(f)
    with open(gates["golden_cases"], encoding="utf-8") as f:
        cases = yaml.safe_load(f)["cases"]
    with open(gates["behaviour_cases"], encoding="utf-8") as f:
        behaviour = yaml.safe_load(f)
    for case in behaviour["invariance"] + behaviour["directional"]:
        case.setdefault("history", behaviour["history"])

    predictor = FraudPredictor(paths["model_dir"])
    predictor.load()
    scorer = RawScorer.from_model_dir(predictor, paths["model_dir"])

    results = (
        check_metrics(metrics, gates["metrics"])
        + check_golden(scorer, cases)
        + check_invariance(scorer, behaviour["invariance"], gates["invariance_tolerance"])
        + check_directional(scorer, behaviour["directional"])
    )
    failed = [r for r in results if not r["passed"]]

    report = {"passed": not failed, "checks": len(results), "failed": len(failed)}
    with open(ensure_parent(paths["gate"]), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if failed:
        logger.error("Quality gate FAILED (%d of %d checks):\n%s",
                     len(failed), len(results), summarise(results))
        raise ModelQualityError(f"{len(failed)} quality gate check(s) failed")
    logger.info("Quality gate passed (%d checks):\n%s", len(results), summarise(results))


if __name__ == "__main__":
    main()