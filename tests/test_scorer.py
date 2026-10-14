"""
Real-time scoring tests.

The central guarantee: scoring a stream of raw log lines one at a time gives
exactly the same fraud probabilities as computing the same features in a
batch replay. If that ever breaks, training and serving have drifted apart.
"""

import copy
import json
import logging

import joblib
import lightgbm as lgb
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from fraudshield.data.generate import generate
from fraudshield.features.engineer import FEATURE_COLS
from fraudshield.features.online import CategoryEncoder, FeatureState
from fraudshield.ingestion.parser import parse_log_line
from fraudshield.serving import api
from fraudshield.serving.predictor import FraudPredictor
from fraudshield.serving.scorer import RawScorer, UnparseableLog

THRESHOLD = 0.5


@pytest.fixture(scope="module")
def stream():
    """Generated raw log lines that parse, in time order, with parsed fields."""
    logging.disable(logging.INFO)
    with open("configs/pipeline.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["generator"]
    rows = []
    for line in generate({**cfg, "n_rows": 3000}):
        parsed = parse_log_line(line)
        if parsed is not None:
            rows.append((line, parsed))
    rows.sort(key=lambda r: (r[1]["timestamp"], r[1]["user_id"]))
    return rows


@pytest.fixture(scope="module")
def bundle(stream, tmp_path_factory):
    """Train a small model on the first 70% of the stream; save a full model bundle."""
    split = int(len(stream) * 0.7)
    history = pd.DataFrame([p for _, p in stream[:split]])
    encoder = CategoryEncoder().fit(history)
    state = FeatureState(encoder)
    X = pd.DataFrame([state.process(p) for _, p in stream[:split]])[FEATURE_COLS].fillna(-1)
    y = (X["amount"] >= 4000).astype(int)
    model = Pipeline([("model", lgb.LGBMClassifier(n_estimators=30, verbosity=-1))]).fit(X, y)

    d = tmp_path_factory.mktemp("bundle")
    joblib.dump(model, d / "fraudshield_model.joblib")
    (d / "best_threshold.txt").write_text(str(THRESHOLD))
    (d / "feature_cols.json").write_text(json.dumps(FEATURE_COLS))
    encoder.save(d / "encoders.json")
    joblib.dump(state, d / "feature_state.joblib")
    return d, split


def make_scorer(model_dir) -> RawScorer:
    predictor = FraudPredictor(model_dir=str(model_dir))
    predictor.load()
    return RawScorer.from_model_dir(predictor, model_dir)


def test_streaming_scores_equal_batch_scores(stream, bundle):
    model_dir, split = bundle
    live = stream[split:]

    scorer = make_scorer(model_dir)
    streamed = [scorer.score(line)["fraud_probability"] for line, _ in live]

    # Batch: the same transactions, features from a copy of the saved state
    batch_scorer = make_scorer(model_dir)
    state = copy.deepcopy(batch_scorer.state)
    features = pd.DataFrame([state.process(p) for _, p in live])
    batch = batch_scorer.predictor.predict(features)["fraud_probability"]

    assert streamed == batch


def test_history_carries_over_between_requests(stream, bundle):
    model_dir, split = bundle
    scorer = make_scorer(model_dir)
    line, parsed = stream[split]
    before = scorer.state.users[parsed["user_id"]].n if parsed["user_id"] in scorer.state.users else 0
    scorer.score(line)
    assert scorer.state.users[parsed["user_id"]].n == before + 1


def test_unparseable_line_is_rejected(bundle):
    scorer = make_scorer(bundle[0])
    with pytest.raises(UnparseableLog):
        scorer.score("MALFORMED_LOG")


@pytest.fixture
def client(bundle, monkeypatch):
    model_dir, _ = bundle
    predictor = FraudPredictor(model_dir=str(model_dir))
    monkeypatch.setattr(api, "predictor", predictor)
    with TestClient(api.app) as c:
        yield c


def test_score_endpoint_returns_a_decision(client, stream, bundle):
    line, parsed = stream[bundle[1]]
    r = client.post("/score", json={"log": line})
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == parsed["user_id"]
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["is_fraud"] == (body["fraud_probability"] >= THRESHOLD)


def test_score_endpoint_rejects_garbage(client):
    r = client.post("/score", json={"log": "this is not a transaction"})
    assert r.status_code == 422