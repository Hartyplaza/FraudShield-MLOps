"""
src/fraudshield/monitoring/drift.py

Drift detection: is the traffic the model sees now still like the traffic it
was checked on?

A reference profile records, for every model feature and for the model's
score, how values were spread over bins on a stream of unseen transactions
scored just after training. A window of recent traffic is binned the same way
and compared with the Population Stability Index (PSI):

    PSI = sum over bins of (current% - reference%) * ln(current% / reference%)

Usual reading: below 0.1 no meaningful shift, 0.1-0.25 moderate, above 0.25 a
major shift. The limits live in configs/pipeline.yaml -> monitoring.

Labels arrive late in fraud (chargebacks take weeks), so drift in the inputs
and in the score is the earliest warning available.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed score bins: the score is a probability, and most traffic sits near 0
SCORE_EDGES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
MAX_CATEGORIES = 12   # a feature with this many distinct values or fewer is binned per value
EPS = 1e-4            # stands in for an empty bin, so ln() stays finite


# ---------------------------------------------------------------- binning
def _numeric_spec(values: pd.Series, bins: int) -> dict:
    present = values.dropna()
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    edges = sorted(set(np.quantile(present, qs).round(6).tolist())) if len(present) else []
    return {"kind": "numeric", "edges": edges}


def _categorical_spec(values: pd.Series) -> dict:
    return {"kind": "categorical", "values": sorted(_key(v) for v in values.dropna().unique())}


def _key(v) -> str:
    """Category label: 1.0 and 1 are the same category."""
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


def proportions(values: pd.Series, spec: dict) -> list[float]:
    """Share of `values` in each bin of `spec`. The last bin is always 'missing';
    for a categorical feature the one before it is 'unseen value'."""
    values = pd.to_numeric(values, errors="coerce")
    present, n = values.dropna(), len(values)
    if spec["kind"] == "numeric":
        idx = np.searchsorted(spec["edges"], present.to_numpy(), side="right")
        counts = np.bincount(idx, minlength=len(spec["edges"]) + 1).tolist()
    else:
        keys = present.map(_key)
        counts = [int((keys == v).sum()) for v in spec["values"]]
        counts.append(int((~keys.isin(spec["values"])).sum()))
    counts.append(n - len(present))
    return [c / n for c in counts] if n else [0.0] * len(counts)


def psi(reference: list[float], current: list[float]) -> float:
    return float(sum((c - r) * math.log(c / r)
                     for r, c in ((max(r, EPS), max(c, EPS)) for r, c in zip(reference, current, strict=True))))


# ---------------------------------------------------------------- reference
def build_reference(features: pd.DataFrame, scores: pd.Series, threshold: float,
                    unparseable_rate: float, bins: int = 10) -> dict:
    """Profile of the traffic the model was checked on."""
    profile = {}
    for col in features.columns:
        values = features[col]
        spec = (_categorical_spec(values) if values.nunique(dropna=True) <= MAX_CATEGORIES
                else _numeric_spec(values, bins))
        profile[col] = {**spec, "props": proportions(values, spec)}
    score_spec = {"kind": "numeric", "edges": SCORE_EDGES}
    return {
        "n": len(features),
        "threshold": threshold,
        "flag_rate": float((scores >= threshold).mean()),
        "unparseable_rate": unparseable_rate,
        "score": {**score_spec, "props": proportions(scores, score_spec)},
        "features": profile,
    }


def save_reference(reference: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(reference, indent=1), encoding="utf-8")


def load_reference(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- report
def _status(value: float, warn: float, alert: float) -> str:
    return "alert" if value > alert else "warn" if value > warn else "ok"


def drift_report(reference: dict, features: pd.DataFrame, scores: pd.Series,
                 limits: dict, unparseable: int = 0) -> dict:
    """Compare a window of recent traffic with the reference.

    `limits` is configs/pipeline.yaml -> monitoring: psi_warn, psi_alert,
    min_samples, max_flag_rate_ratio, max_unparseable_increase, and
    report_only (features whose PSI is shown but never raises a warning or alert).
    """
    n = len(features)
    total = n + unparseable
    report = {"n": n, "unparseable": unparseable}
    if n < limits["min_samples"]:
        return {**report, "status": "insufficient_data",
                "detail": f"{n} scored transactions; at least {limits['min_samples']} needed"}

    warn, alert = limits["psi_warn"], limits["psi_alert"]
    report_only = set(limits.get("report_only", []))
    feats = {}
    for col, ref in reference["features"].items():
        value = psi(ref["props"], proportions(features[col], ref)) if col in features else float("inf")
        status = "info" if col in report_only else _status(value, warn, alert)
        feats[col] = {"psi": round(value, 4), "status": status}

    score_psi = psi(reference["score"]["props"], proportions(scores, reference["score"]))
    flag_rate = float((scores >= reference["threshold"]).mean())
    ref_flag = max(reference["flag_rate"], EPS)
    unparseable_rate = unparseable / total if total else 0.0

    checks = {
        "score_psi": {"value": round(score_psi, 4), "status": _status(score_psi, warn, alert)},
        "flag_rate": {"value": round(flag_rate, 4), "reference": round(reference["flag_rate"], 4),
                      "status": "alert" if (flag_rate / ref_flag > limits["max_flag_rate_ratio"]
                                            or ref_flag / max(flag_rate, EPS) > limits["max_flag_rate_ratio"])
                      else "ok"},
        "unparseable_rate": {"value": round(unparseable_rate, 4),
                             "reference": round(reference["unparseable_rate"], 4),
                             "status": "alert" if unparseable_rate - reference["unparseable_rate"]
                             > limits["max_unparseable_increase"] else "ok"},
    }
    statuses = [f["status"] for f in feats.values()] + [c["status"] for c in checks.values()]
    status = "alert" if "alert" in statuses else "warn" if "warn" in statuses else "ok"
    ranked = sorted(feats.items(), key=lambda kv: -kv[1]["psi"])
    return {
        **report,
        "status": status,
        "alerts": [c for c, f in ranked if f["status"] == "alert"],
        "warnings": [c for c, f in ranked if f["status"] == "warn"],
        "checks": checks,
        "features": dict(ranked),
    }
