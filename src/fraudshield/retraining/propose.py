"""
src/fraudshield/retraining/propose.py

Turn a retraining decision into a reviewable change.

Labels for the two weeks after the training data arrive (here simulated: the
period is generated, optionally with a drift scenario, and labelled by the
labelling rules). The decision rule (retrain.py) runs on them. If it says
`retrained`, --apply edits configs/pipeline.yaml so that the pipeline itself
retrains, and the change goes through the normal pull request and CI:

    1. the period is appended to retraining.periods (training data);
    2. the scenario's changes become part of normal traffic (monitoring.normal):
       the drift reference, the drills and the live demo follow the new normal,
       so the monitor stops alerting on what was deliberately accepted;
    3. if the period breaks the data contract's amount range, the range is
       widened, and the report says so: accepting a new normal is a contract
       change, for a person to approve.

Any other contract violation, or a `recalibrated` decision, is reported and
not applied.

    uv run python -m fraudshield.retraining.propose --scenario amounts_x3 [--apply]
"""

import argparse
import json
import logging
import math
import os
import re
from pathlib import Path

import joblib
import pandas as pd
import yaml

from fraudshield.pipeline.config import (
    ensure_parent,
    load_config,
    normal_traffic,
    training_end,
    training_periods,
)
from fraudshield.retraining.retrain import assess, build_split, decide, labelled, period
from fraudshield.validation.schemas import (
    DataValidationError,
    labelled_schema,
    validate,
)

logger = logging.getLogger(__name__)

# 30 days: the longest feature window (txn_count_30d, txn_sum_30d). A shorter
# period leaves those features in transition after a new normal is accepted, and
# the drift reference cannot settle. 5360 rows keeps the usual rate (2500 per 14 days).
PERIOD_DAYS, PERIOD_ROWS, FIRST_SEED = 30, 5360, 100


def recent_period(cfg: dict, scenario: str) -> dict:
    """Generator settings for the labelled period after the training data."""
    changes = {} if scenario == "none" else cfg["monitoring"]["drill"]["scenarios"][scenario]
    seed = FIRST_SEED + len(cfg["retraining"]["periods"])
    # Stored in full (today's normal plus the scenario), so the period regenerates
    # the same data whatever normal becomes later
    return {"seed": seed, "start": training_end(cfg), "days": PERIOD_DAYS, "n_rows": PERIOD_ROWS,
            **cfg["monitoring"].get("normal", {}), **changes}


def contract_changes(cfg: dict, data: pd.DataFrame) -> tuple[dict, list[str]]:
    """(changes this script can make, violations it cannot) for the new period."""
    try:
        validate(data, labelled_schema(cfg["validation"]), "retraining")
        return {}, []
    except DataValidationError as e:
        summary = e.summary
    changes, other = {}, []
    for _, row in summary.iterrows():
        if row["column"] == "amount" and row["check"].startswith("in_range"):
            lo, hi = cfg["validation"]["amount_range"]
            new_hi = max(hi, math.ceil(data["amount"].max() / 5000) * 5000)
            changes["amount_range"] = [lo, new_hi]
        else:
            other.append(f"{row['column']}: {row['check']}")
    return changes, other


def flow(d: dict) -> str:
    return yaml.safe_dump(d, default_flow_style=True, width=10_000, sort_keys=False).strip()


def apply(config_path: str, cfg: dict, new_period: dict, scenario: str, contract: dict) -> list[str]:
    """Edit the config text in place (comments kept). Returns what changed."""
    p = Path(config_path)
    text = p.read_text(encoding="utf-8")
    done = []

    item = f"    - {flow(new_period)}\n"
    if "  periods: []\n" in text:
        text = text.replace("  periods: []\n", "  periods:\n" + item, 1)
    else:
        m = re.search(r"^  periods:\n(?:    - .*\n)*", text, flags=re.MULTILINE)
        text = text[:m.end()] + item + text[m.end():]
    done.append(f"training data: added period {flow(new_period)}")

    if scenario != "none":
        normal = {**cfg["monitoring"].get("normal", {}), **cfg["monitoring"]["drill"]["scenarios"][scenario]}
        text, n = re.subn(r"^  normal: .*\n", f"  normal: {flow(normal)}\n", text, count=1, flags=re.MULTILINE)
        if not n:
            raise SystemExit("monitoring.normal not found in the config")
        done.append(f"NEW NORMAL: monitoring.normal = {flow(normal)}; the drift reference and drills follow it")

    if "amount_range" in contract:
        lo, hi = contract["amount_range"]
        text = re.sub(r"^(  amount_range: )\[.*\]$", rf"\g<1>[{lo}, {hi}]", text, count=1, flags=re.MULTILINE)
        done.append(f"DATA CONTRACT: amount_range widened to [{lo}, {hi}] (review this)")

    p.write_text(text, encoding="utf-8")
    return done


def markdown(report: dict) -> str:
    lines = [f"## Retraining proposal: {report['decision']}", "",
             f"Recent labelled period: `{flow(report['period'])}` (scenario `{report['scenario']}`)", "",
             f"**Reason:** {report['reason']}", "",
             "| Model | Threshold | Recent recall | Recent precision | Original recall | Original precision |",
             "|---|---|---|---|---|---|"]
    for name, c in report["candidates"].items():
        lines.append(f"| {name} | {c['threshold']:.2f} | {c['recent']['recall']:.3f} | "
                     f"{c['recent']['precision']:.3f} | {c['base']['recall']:.3f} | {c['base']['precision']:.3f} |")
    if report["contract_changes"] or report["contract_violations"]:
        lines += ["", (f"**Data contract:** changes {report['contract_changes'] or 'none'}; "
                       f"not handled {report['contract_violations'] or 'none'}")]
    if report.get("applied"):
        lines += ["", "**Applied to configs/pipeline.yaml:**", *[f"- {a}" for a in report["applied"]]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide on retraining with newly labelled traffic.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--scenario", default="none", help="drift scenario of the labelled period, or none")
    parser.add_argument("--apply", action="store_true", help="edit the config if the decision is `retrained`")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    cfg = load_config(args.config)
    paths, rt, gates = cfg["paths"], cfg["retraining"], cfg["gates"]["metrics"]
    model_dir = Path(paths["model_dir"])
    champion = joblib.load(model_dir / "fraudshield_model.joblib")
    threshold = float((model_dir / "best_threshold.txt").read_text(encoding="utf-8"))
    feature_cols = json.loads((model_dir / "feature_cols.json").read_text(encoding="utf-8"))

    new_period = recent_period(cfg, args.scenario)
    logging.getLogger("fraudshield").setLevel(logging.WARNING)
    base = pd.concat([period(p) for p in training_periods(cfg)], ignore_index=True)
    recent = period({**normal_traffic(cfg), **new_period})
    # The pipeline validates all training data together, so check the contract the same way
    contract, violations = contract_changes(cfg, labelled(pd.concat([base, recent], ignore_index=True)))
    split = build_split(base, recent, feature_cols, rt["holdout_days"])
    results = assess(champion, threshold, split, cfg["training"]["min_precision"])
    logging.getLogger("fraudshield").setLevel(logging.INFO)
    decision, reason = decide(results, {
        "min_recall": gates["min_recall"], "min_precision": gates["min_precision"],
        "min_holdout_frauds": rt["min_holdout_frauds"], "max_base_pr_auc_drop": rt["max_base_pr_auc_drop"]})

    report = {"decision": decision, "reason": reason, "scenario": args.scenario, "period": new_period,
              "candidates": {n: {"threshold": r["threshold"], "recent": r["recent"], "base": r["base"]}
                             for n, r in results.items()},
              "contract_changes": contract, "contract_violations": violations, "applied": []}

    if args.apply and decision == "retrained":
        if violations:
            report["reason"] += f"; NOT applied: data contract violations need a person: {violations}"
        else:
            report["applied"] = apply(args.config, cfg, new_period, args.scenario, contract)
    elif args.apply and decision == "recalibrated":
        report["reason"] += "; NOT applied: threshold-only changes are not automated"

    out = ensure_parent("reports/retrain_decision.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = markdown(report)
    print(md)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md)
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"decision={decision}\napplied={'true' if report['applied'] else 'false'}\n")


if __name__ == "__main__":
    main()
