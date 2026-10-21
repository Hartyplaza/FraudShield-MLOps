"""
Stage 6: seed robustness. Re-runs the model recipe (the chosen model with its
tuned hyperparameters, and the threshold rule) on several freshly generated
datasets, and reports each metric as mean, spread and worst case.

One test set holds only ~60 frauds, so a single run cannot tell a real
improvement from luck. Comparing a change against this spread can.

No hyperparameter search here: the question is how much the results move with
the data, not whether a new search finds different settings.

    uv run python -m fraudshield.pipeline.seeds
"""

import json
import logging

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split

from fraudshield.data.generate import generate
from fraudshield.evaluation.metrics import evaluate
from fraudshield.evaluation.threshold_tuner import tune_threshold_for_precision
from fraudshield.features.online import CategoryEncoder, replay
from fraudshield.ingestion.cleaner import clean
from fraudshield.ingestion.parser import parse_logs
from fraudshield.labelling.label_simulator import simulate_labels
from fraudshield.pipeline.config import ensure_parent, stage_args

logger = logging.getLogger(__name__)

SPLIT_SEED = 42
REPORTED = ["pr_auc", "recall", "precision", "fpr", "threshold"]


def run_seed(cfg: dict, seed: int, recipe, feature_cols: list[str]) -> dict:
    """Generate a dataset with `seed`, then fit, tune and evaluate the recipe on it."""
    raw = pd.DataFrame({"raw_log": generate({**cfg["generator"], "seed": seed})})
    df = simulate_labels(clean(parse_logs(raw)))
    features, _ = replay(df, CategoryEncoder().fit(df))
    X, y = features[feature_cols].fillna(-1), df["is_fraud"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SPLIT_SEED)
    X_train, _, y_train, _ = train_test_split(
        X_trainval, y_trainval, test_size=0.20, stratify=y_trainval, random_state=SPLIT_SEED)

    model = clone(recipe).fit(X_train, y_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    threshold = tune_threshold_for_precision(
        model, X_trainval, y_trainval, cfg["training"]["min_precision"], cv)
    metrics = evaluate(model, X_test, y_test, threshold=threshold)
    return {"seed": seed, "test_frauds": int(y_test.sum()), **metrics}


def main() -> None:
    cfg = stage_args("Evaluate the model recipe across several generated datasets.")
    paths = cfg["paths"]

    recipe = joblib.load(f"{paths['model_dir']}/fraudshield_model.joblib")
    with open(f"{paths['model_dir']}/feature_cols.json", encoding="utf-8") as f:
        feature_cols = json.load(f)

    logging.getLogger("fraudshield").setLevel(logging.WARNING)
    runs = pd.DataFrame([run_seed(cfg, s, recipe, feature_cols) for s in cfg["seeds"]["seeds"]])
    logging.getLogger("fraudshield").setLevel(logging.INFO)

    summary = runs[REPORTED].agg(["mean", "std", "min", "max"]).round(4)
    report = {
        "seeds": len(runs),
        **{f"{m}_mean": float(summary.loc["mean", m]) for m in REPORTED},
        **{f"{m}_std": float(summary.loc["std", m]) for m in REPORTED},
        "pr_auc_min": float(summary.loc["min", "pr_auc"]),
        "recall_min": float(summary.loc["min", "recall"]),
        "precision_min": float(summary.loc["min", "precision"]),
    }
    with open(ensure_parent(paths["seeds"]), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cols = ["seed", "test_frauds", *REPORTED, "tp", "fp", "fn"]
    logger.info("Per seed:\n%s", runs[cols].round(4).to_string(index=False))
    logger.info("Across %d seeds:\n%s", len(runs), summary.to_string())


if __name__ == "__main__":
    main()