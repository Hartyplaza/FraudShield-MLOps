"""Model registry: a bundle is registered as a pyfunc model, versioned, aliased,
and scores exactly like the API when loaded back by name."""

import json
import logging

import joblib
import lightgbm as lgb
import mlflow
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fraudshield.data.generate import generate
from fraudshield.features.engineer import FEATURE_COLS
from fraudshield.features.online import CategoryEncoder, FeatureState
from fraudshield.ingestion.parser import parse_log_line
from fraudshield.pipeline.config import load_config
from fraudshield.registry.promote import (
    CHAMPION,
    compare,
    current_version,
    decide,
    set_champion,
)
from fraudshield.registry.register import CHALLENGER, register_bundle
from fraudshield.serving.predictor import FraudPredictor
from fraudshield.serving.scorer import RawScorer


@pytest.fixture(scope="module")
def lines():
    logging.disable(logging.INFO)
    gen = {**load_config()["generator"], "n_rows": 2000}
    parsed = [(line, parse_log_line(line)) for line in generate(gen)]
    parsed = sorted([p for p in parsed if p[1]], key=lambda r: (r[1]["timestamp"], r[1]["user_id"]))
    return parsed


@pytest.fixture(scope="module")
def bundle(lines, tmp_path_factory):
    """A small but complete model bundle trained on the first 80% of the lines."""
    split = int(len(lines) * 0.8)
    encoder = CategoryEncoder().fit(pd.DataFrame([p for _, p in lines[:split]]))
    state = FeatureState(encoder)
    X = pd.DataFrame([state.process(p) for _, p in lines[:split]])[FEATURE_COLS].fillna(-1)
    y = (X["amount"] >= 4000).astype(int)
    model = Pipeline([("model", lgb.LGBMClassifier(n_estimators=20, verbosity=-1))]).fit(X, y)

    d = tmp_path_factory.mktemp("bundle")
    joblib.dump(model, d / "fraudshield_model.joblib")
    (d / "best_threshold.txt").write_text("0.5")
    (d / "feature_cols.json").write_text(json.dumps(FEATURE_COLS))
    encoder.save(d / "encoders.json")
    joblib.dump(state, d / "feature_state.joblib")
    return d, [line for line, _ in lines[split:split + 50]]


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    """A throwaway MLflow store, so tests never touch the real one."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test")
    yield mlflow.MlflowClient()
    mlflow.set_tracking_uri(None)


def new_run() -> str:
    with mlflow.start_run() as run:
        return run.info.run_id


def test_each_registration_is_a_new_version_and_becomes_challenger(bundle, tracking):
    model_dir, _ = bundle
    v1 = register_bundle(str(model_dir), new_run(), "fs", {"gate": "passed"})
    v2 = register_bundle(str(model_dir), new_run(), "fs", {"gate": "passed"})
    assert (v1, v2) == (1, 2)
    challenger = tracking.get_model_version_by_alias("fs", CHALLENGER)
    assert int(challenger.version) == 2 and challenger.tags["gate"] == "passed"


def test_registered_model_scores_like_the_api(bundle, tracking):
    model_dir, stream = bundle
    register_bundle(str(model_dir), new_run(), "fs", {})

    loaded = mlflow.pyfunc.load_model(f"models:/fs@{CHALLENGER}")
    via_registry = loaded.predict(pd.DataFrame({"raw_log": stream}))["fraud_probability"].tolist()

    predictor = FraudPredictor(str(model_dir))
    predictor.load()
    scorer = RawScorer.from_model_dir(predictor, model_dir)
    direct = [scorer.score(line)["fraud_probability"] for line in stream]
    assert via_registry == direct


# ---------------------------------------------------------------- promotion
def test_decide_allows_a_small_drop_but_not_a_large_one():
    close = pd.DataFrame({"diff": [-0.004, 0.002, -0.006, 0.000]})
    worse = pd.DataFrame({"diff": [-0.05, -0.02, -0.08, 0.01]})
    assert decide(close, max_drop=0.005)[0]
    promote, summary = decide(worse, max_drop=0.005)
    assert not promote and "better on 1 of 4" in summary


def test_set_champion_moves_the_alias_and_logs_every_move(bundle, tracking, tmp_path):
    model_dir, _ = bundle
    log = tmp_path / "promotions.jsonl"
    v1 = register_bundle(str(model_dir), new_run(), "fs", {})
    v2 = register_bundle(str(model_dir), new_run(), "fs", {})
    set_champion(tracking, "fs", v1, "first", str(log))
    set_champion(tracking, "fs", v2, "better", str(log))
    set_champion(tracking, "fs", v1, "rollback", str(log))
    assert current_version(tracking, "fs", CHAMPION) == v1
    assert tracking.get_model_version("fs", str(v1)).tags["previous_champion"] == str(v2)
    moves = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(m["from"], m["to"], m["reason"]) for m in moves] == [
        (None, v1, "first"), (v1, v2, "better"), (v2, v1, "rollback")]


def test_compare_scores_both_versions_on_the_same_datasets(bundle, tracking):
    model_dir, _ = bundle
    v1 = register_bundle(str(model_dir), new_run(), "fs", {})
    cfg = load_config()
    cfg["generator"]["n_rows"] = 2000
    cfg["seeds"]["seeds"] = [3]
    table = compare(cfg, "fs", v1, v1)
    # Same version on the same data: the paired difference is exactly zero
    assert len(table) == 1 and table["diff"].iloc[0] == 0