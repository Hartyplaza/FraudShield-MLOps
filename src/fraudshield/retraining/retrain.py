"""
src/fraudshield/retraining/retrain.py

Decide whether, and how, to update the model once labels for recent traffic
have arrived (in fraud, chargebacks confirm labels weeks later).

    1. Harm: is the current model (champion) actually worse on recent labelled
       traffic? Drift alone is not enough: most drift does not hurt the model.
       With a few dozen labelled frauds the estimates are noisy, so harm needs
       the 95% upper bound of recall or precision to be below the gate's limit.
    2. Candidates, cheapest first:
         recalibrated: the champion with its threshold re-tuned on recent data
         retrained:    the same recipe refitted on the original + recent data
    3. Accept the first candidate that meets the gate's recall and precision
       limits on a recent holdout (the last days, never trained on), is not
       clearly below them on the original test set (95% upper bound, the same
       standard of evidence as for harm), and is not much worse there in PR-AUC.
       If none does, keep the champion and leave the call to a person.

The original test rows are split exactly as in training, so neither model has
seen them; the recent holdout is the latest `holdout_days`, so the decision is
made on the most recent traffic, as a real loop would have to.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from fraudshield.data.generate import generate
from fraudshield.evaluation.metrics import evaluate
from fraudshield.features.online import CategoryEncoder, FeatureState, replay
from fraudshield.ingestion.cleaner import clean
from fraudshield.ingestion.parser import parse_logs
from fraudshield.labelling.label_simulator import simulate_labels

logger = logging.getLogger(__name__)

SPLIT_SEED = 42   # the training split (training/train.py)


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the labelling rules in each user's time order; keep the row order."""
    ordered = df.sort_values(["user_id", "timestamp"], kind="stable").copy()
    return simulate_labels(ordered).loc[df.index]


def period(generator_cfg: dict) -> pd.DataFrame:
    """Parsed and cleaned transactions for one generated period (no labels)."""
    raw = pd.DataFrame({"raw_log": generate(generator_cfg)})
    return clean(parse_logs(raw))


@dataclass
class Split:
    X: pd.DataFrame
    y: pd.Series
    state: FeatureState          # every user's history after the recent period
    base_trainval: np.ndarray    # positions trained on originally
    base_test: np.ndarray        # original test rows: never trained on
    recent_train: np.ndarray
    recent_holdout: np.ndarray   # last holdout_days of the recent period


def build_split(base: pd.DataFrame, recent: pd.DataFrame, feature_cols: list[str],
                holdout_days: int) -> Split:
    data = labelled(pd.concat([base, recent], ignore_index=True))
    features, state = replay(data, CategoryEncoder().fit(data))
    X, y = features[feature_cols].fillna(-1), data["is_fraud"]

    n_base = len(base)
    base_idx = np.arange(n_base)
    base_trainval, base_test = train_test_split(
        base_idx, test_size=0.20, stratify=y.iloc[:n_base], random_state=SPLIT_SEED)

    recent_idx = np.arange(n_base, len(data))
    times = data["timestamp"].iloc[recent_idx]
    cut = times.max().normalize() - pd.Timedelta(days=holdout_days - 1)
    held = (times >= cut).to_numpy()
    return Split(X, y, state, base_trainval, base_test, recent_idx[~held], recent_idx[held])


def lowest_threshold(p: np.ndarray, y: np.ndarray, min_precision: float, step: float = 0.01) -> float:
    """Lowest threshold (highest recall) at which precision meets the target."""
    for t in np.round(np.arange(step, 1.0, step), 2):
        flagged = p >= t
        if flagged.any() and y[flagged].mean() >= min_precision:
            return float(t)
    return 0.99


def threshold_on(model, X: pd.DataFrame, y: pd.Series, min_precision: float) -> float:
    """Lowest threshold whose precision on (X, y) meets the target, for a fitted model."""
    return lowest_threshold(model.predict_proba(X)[:, 1], np.asarray(y), min_precision)


def candidates(champion, split: Split, min_precision: float) -> dict:
    """name -> (model, threshold), cheapest first."""
    recent_X, recent_y = split.X.iloc[split.recent_train], split.y.iloc[split.recent_train]
    recalibrated = threshold_on(champion, recent_X, recent_y, min_precision)

    train_idx = np.concatenate([split.base_trainval, split.recent_train])
    X_tv, y_tv = split.X.iloc[train_idx], split.y.iloc[train_idx]
    X_fit, _, y_fit, _ = train_test_split(X_tv, y_tv, test_size=0.20, stratify=y_tv, random_state=SPLIT_SEED)
    model = clone(champion).fit(X_fit, y_fit)
    # Out-of-fold scores; the threshold must meet the precision target on the
    # original and on the recent traffic separately, not just on their mix
    # (where a high-fraud recent period would dominate).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    oof = cross_val_predict(clone(champion), X_tv, y_tv, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    is_base = np.r_[np.ones(len(split.base_trainval), bool), np.zeros(len(split.recent_train), bool)]
    y_arr = y_tv.to_numpy()
    threshold = max(lowest_threshold(oof[is_base], y_arr[is_base], min_precision),
                    lowest_threshold(oof[~is_base], y_arr[~is_base], min_precision))
    return {"recalibrated": (champion, recalibrated), "retrained": (model, threshold)}


def scores(model, threshold: float, X: pd.DataFrame, y: pd.Series) -> dict:
    m = evaluate(model, X, y, threshold=threshold)
    return {k: m[k] for k in ("pr_auc", "recall", "precision", "tp", "fp", "fn")} | {"frauds": int(y.sum())}


def wilson_upper(successes: int, n: int, z: float = 1.96) -> float:
    """Upper end of the 95% Wilson interval for a proportion (1.0 when n is 0)."""
    if n == 0:
        return 1.0
    p = successes / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + margin) / (1 + z * z / n))


def assess(champion, champion_threshold: float, split: Split, min_precision: float) -> dict:
    """Every model's scores on the recent holdout and the original test set."""
    hold = (split.X.iloc[split.recent_holdout], split.y.iloc[split.recent_holdout])
    base = (split.X.iloc[split.base_test], split.y.iloc[split.base_test])
    models = {"champion": (champion, champion_threshold), **candidates(champion, split, min_precision)}
    return {name: {"threshold": t, "recent": scores(m, t, *hold), "base": scores(m, t, *base),
                   "model": m}
            for name, (m, t) in models.items()}


def decide(results: dict, limits: dict) -> tuple[str, str]:
    """(decision, reason). decision is 'keep' or a candidate's name.

    limits: min_recall, min_precision (the gate's), min_holdout_frauds,
    max_base_pr_auc_drop.
    """
    champ = results["champion"]
    frauds = champ["recent"]["frauds"]
    if frauds < limits["min_holdout_frauds"]:
        return "keep", f"only {frauds} labelled frauds in the holdout; need {limits['min_holdout_frauds']}"

    def meets(s: dict) -> bool:
        return s["recall"] >= limits["min_recall"] and s["precision"] >= limits["min_precision"]

    r = champ["recent"]
    recall_up = wilson_upper(r["tp"], r["tp"] + r["fn"])
    precision_up = wilson_upper(r["tp"], r["tp"] + r["fp"])
    if recall_up >= limits["min_recall"] and precision_up >= limits["min_precision"]:
        return "keep", (f"no clear harm: champion recall {r['recall']:.3f} (up to {recall_up:.3f}), "
                        f"precision {r['precision']:.3f} (up to {precision_up:.3f}) on recent traffic")

    def clearly_below(s: dict) -> bool:
        return (wilson_upper(s["tp"], s["tp"] + s["fn"]) < limits["min_recall"]
                or wilson_upper(s["tp"], s["tp"] + s["fp"]) < limits["min_precision"])

    failures = []
    for name, res in results.items():
        if name == "champion":
            continue
        drop = champ["base"]["pr_auc"] - res["base"]["pr_auc"]
        problems = []
        if not meets(res["recent"]):
            problems.append(f"recent recall {res['recent']['recall']:.3f} / precision {res['recent']['precision']:.3f}")
        if clearly_below(res["base"]):
            problems.append(f"original-traffic recall {res['base']['recall']:.3f} / precision {res['base']['precision']:.3f}")
        if drop > limits["max_base_pr_auc_drop"]:
            problems.append(f"original-traffic PR-AUC down {drop:.3f}")
        if not problems:
            c = res["recent"]
            return name, (f"harm: champion recall {r['recall']:.3f} (at most {recall_up:.3f}), "
                          f"precision {r['precision']:.3f} (at most {precision_up:.3f}) on recent traffic; "
                          f"{name} recall {c['recall']:.3f}, precision {c['precision']:.3f}, "
                          f"and is not clearly below the limits on original traffic"
                          + (f"; rejected first: {' | '.join(failures)}" if failures else ""))
        failures.append(f"{name}: " + "; ".join(problems))
    return "keep", "harm, but no candidate meets the limits (" + " | ".join(failures) + "): needs a person"
