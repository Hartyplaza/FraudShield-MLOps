"""Retraining decision rule: harm needs clear evidence; candidates must hold on old and new traffic."""

import numpy as np
import pandas as pd
import pytest

from fraudshield.pipeline.config import load_config
from fraudshield.retraining.retrain import decide, threshold_on, wilson_upper

CFG = load_config()
LIMITS = {
    "min_recall": CFG["gates"]["metrics"]["min_recall"],
    "min_precision": CFG["gates"]["metrics"]["min_precision"],
    "min_holdout_frauds": CFG["retraining"]["min_holdout_frauds"],
    "max_base_pr_auc_drop": CFG["retraining"]["max_base_pr_auc_drop"],
}


def s(tp: int, fp: int, fn: int, pr_auc: float = 0.99) -> dict:
    return {"pr_auc": pr_auc, "tp": tp, "fp": fp, "fn": fn, "frauds": tp + fn,
            "recall": tp / (tp + fn), "precision": tp / (tp + fp) if tp + fp else 0.0}


GOOD_BASE = s(59, 6, 2, pr_auc=0.993)


def results(champion_recent: dict, **cands) -> dict:
    out = {"champion": {"recent": champion_recent, "base": GOOD_BASE}}
    for name, (recent, base) in cands.items():
        out[name] = {"recent": recent, "base": base}
    return out


def test_wilson_upper_bound():
    assert wilson_upper(0, 0) == 1.0
    assert wilson_upper(10, 10) == pytest.approx(1.0)
    assert 0.80 < wilson_upper(16, 20) < 0.95          # 0.80 measured on 20 cases could be ~0.92
    assert wilson_upper(160, 200) < wilson_upper(16, 20)  # more evidence, tighter bound


def test_no_harm_keeps_the_champion():
    decision, reason = decide(results(s(60, 5, 3)), LIMITS)
    assert decision == "keep" and "no clear harm" in reason


def test_a_noisy_dip_on_few_frauds_is_not_harm():
    # 16 of 20 alerts real: precision 0.80 < 0.85, but 20 alerts cannot rule out 0.85
    decision, reason = decide(results(s(16, 4, 1)), LIMITS)
    assert decision == "keep" and "no clear harm" in reason


def test_too_few_labelled_frauds_keeps_the_champion():
    decision, reason = decide(results(s(3, 0, 7)), LIMITS)
    assert decision == "keep" and "labelled frauds" in reason


def test_clear_harm_takes_the_cheapest_candidate_that_holds_everywhere():
    harmed = s(47, 0, 19)                                         # recall 0.71 on 66 frauds
    decision, reason = decide(results(harmed,
                                      recalibrated=(s(64, 7, 2), s(61, 35, 0)),   # floods old traffic
                                      retrained=(s(66, 9, 0), s(61, 9, 0))), LIMITS)
    assert decision == "retrained"
    assert "rejected first: recalibrated" in reason


def test_a_candidate_much_worse_on_old_traffic_is_rejected():
    harmed = s(47, 0, 19)
    decision, reason = decide(results(harmed, retrained=(s(66, 9, 0), s(61, 9, 0, pr_auc=0.95))), LIMITS)
    assert decision == "keep" and "needs a person" in reason


def test_threshold_on_meets_the_precision_target():
    class Fixed:
        def predict_proba(self, X):
            p = X["p"].to_numpy()
            return np.column_stack([1 - p, p])

    X = pd.DataFrame({"p": [0.95, 0.9, 0.8, 0.7, 0.3, 0.2]})
    y = pd.Series([1, 1, 1, 0, 1, 0])
    t = threshold_on(Fixed(), X, y, min_precision=0.9)
    flagged = X["p"] >= t
    assert y[flagged].mean() >= 0.9 and t == pytest.approx(0.71)


def test_drill_expectations_name_known_scenarios():
    known = {"none", *CFG["monitoring"]["drill"]["scenarios"]}
    assert set(CFG["retraining"]["drill"]["expect"]) <= known
    assert "retrained" in CFG["retraining"]["drill"]["expect"].values()
