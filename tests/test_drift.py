"""Drift monitor: binning, PSI, and the report's alert logic, on small synthetic data."""

import numpy as np
import pandas as pd
import pytest

from fraudshield.features.engineer import FEATURE_COLS
from fraudshield.monitoring.drift import build_reference, drift_report, proportions, psi
from fraudshield.pipeline.config import load_config

MON = load_config()["monitoring"]
LIMITS = {**MON, "min_samples": 100}


def traffic(n: int, seed: int, amount_scale: float = 1.0, device_codes=(0, 1, 2, 3)) -> tuple:
    rng = np.random.default_rng(seed)
    features = pd.DataFrame({
        "amount": rng.uniform(1, 5000, n) * amount_scale,
        "device_encoded": rng.choice(device_codes, n),
        "user_mean_amount": rng.normal(2500, 100, n),
    })
    scores = pd.Series(rng.beta(0.3, 6, n))
    return features, scores


@pytest.fixture(scope="module")
def reference():
    features, scores = traffic(5000, seed=0)
    return build_reference(features, scores, threshold=0.5, unparseable_rate=0.1)


def test_identical_distributions_have_zero_psi():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0)


def test_psi_grows_with_the_shift():
    ref = [0.25, 0.25, 0.25, 0.25]
    assert psi(ref, [0.3, 0.2, 0.25, 0.25]) < psi(ref, [0.5, 0.1, 0.2, 0.2]) < psi(ref, [0.97, 0.01, 0.01, 0.01])


def test_proportions_cover_missing_and_unseen_values():
    spec = {"kind": "categorical", "values": ["0", "1"]}
    props = proportions(pd.Series([0, 1, 1, 7, None]), spec)   # bins: 0, 1, unseen, missing
    assert props == pytest.approx([0.2, 0.4, 0.2, 0.2])
    numeric = proportions(pd.Series([1.0, 5.0, 9.0, np.nan]), {"kind": "numeric", "edges": [4.0, 8.0]})
    assert numeric == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_reference_bins_numeric_by_quantile_and_codes_by_value(reference):
    assert reference["features"]["amount"]["kind"] == "numeric"
    assert len(reference["features"]["amount"]["edges"]) == 9                      # deciles
    assert reference["features"]["device_encoded"]["kind"] == "categorical"
    assert sum(reference["features"]["amount"]["props"]) == pytest.approx(1.0)


def test_same_traffic_is_ok(reference):
    report = drift_report(reference, *traffic(1000, seed=1), LIMITS)
    assert report["status"] == "ok", report["alerts"]


def test_shifted_amounts_raise_an_alert(reference):
    report = drift_report(reference, *traffic(1000, seed=1, amount_scale=3.0), LIMITS)
    assert report["status"] == "alert"
    assert report["alerts"][0] == "amount"


def test_unseen_category_codes_raise_an_alert(reference):
    report = drift_report(reference, *traffic(1000, seed=1, device_codes=(-1,)), LIMITS)
    assert "device_encoded" in report["alerts"]


def test_report_only_features_never_alert(reference):
    features, scores = traffic(1000, seed=1)
    features["user_mean_amount"] += 5000                                           # huge shift
    report = drift_report(reference, features, scores, {**LIMITS, "report_only": ["user_mean_amount"]})
    assert report["features"]["user_mean_amount"]["status"] == "info"
    assert report["status"] == "ok"


def test_flag_rate_and_unparseable_checks(reference):
    features, scores = traffic(1000, seed=1)
    report = drift_report(reference, features, scores.clip(lower=0.6), LIMITS)    # everything flagged
    assert report["checks"]["flag_rate"]["status"] == "alert"
    report = drift_report(reference, features, scores, LIMITS, unparseable=1000)  # half the lines broken
    assert report["checks"]["unparseable_rate"]["status"] == "alert"


def test_too_little_traffic_is_not_judged(reference):
    report = drift_report(reference, *traffic(50, seed=1), LIMITS)
    assert report["status"] == "insufficient_data"


def test_monitoring_config_names_real_features():
    assert set(MON["report_only"]) <= set(FEATURE_COLS)
    assert MON["min_samples"] <= MON["window"]
    assert MON["psi_warn"] < MON["psi_alert"]
