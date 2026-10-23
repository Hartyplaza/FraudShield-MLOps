"""
API tests: call the real FastAPI app in-process with a small model trained
on the fly, so the tests do not depend on `make all` having been run.
"""

import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from fraudshield.features.engineer import FEATURE_COLS
from fraudshield.serving import api
from fraudshield.serving.predictor import FraudPredictor

THRESHOLD = 0.5

VALID_TXN = {
    "hour": 2, "day_of_week": 5, "is_night": 1, "is_weekend": 1,
    "is_business_hours": 0, "amount": 4800.0, "amount_z_score": 3.2,
    "is_high_amount": 1, "amount_vs_user_max": 0.96,
    "user_mean_amount": 2400.0, "user_std_amount": 1200.0,
    "user_max_amount": 4997.0, "txn_count_1d": 1, "txn_sum_1d": 4800.0,
    "txn_count_7d": 5, "txn_sum_7d": 12000.0, "txn_count_30d": 18,
    "txn_sum_30d": 43000.0, "device_changed": 1, "location_changed": 0,
    "time_since_last_txn_hrs": 0.5, "location_missing": 1,
    "device_missing": 0, "txn_type_encoded": 0, "location_encoded": 3,
    "device_encoded": 1, "log_format_encoded": 2, "currency_encoded": 1,
}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """Train a tiny model and write the same artefacts the pipeline writes."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((300, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y = (X["amount"] > 0.8).astype(int)
    model = Pipeline([("model", lgb.LGBMClassifier(n_estimators=20, verbosity=-1))])
    model.fit(X, y)

    d = tmp_path_factory.mktemp("models")
    joblib.dump(model, d / "fraudshield_model.joblib")
    (d / "best_threshold.txt").write_text(str(THRESHOLD))
    (d / "feature_cols.json").write_text(json.dumps(FEATURE_COLS))
    return d


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setattr(api, "predictor", FraudPredictor(model_dir=str(model_dir)))
    with TestClient(api.app) as c:
        yield c


def test_health_reports_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["threshold"] == THRESHOLD


def test_predict_returns_a_consistent_decision(client):
    r = client.post("/predict", json=VALID_TXN)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["is_fraud"] == (body["fraud_probability"] >= body["threshold_used"])
    assert body["risk_level"] in {"low", "medium", "high", "critical"}
    assert len(body["top_features"]) == 5
    assert set(body["top_features"]) <= set(FEATURE_COLS)


def test_invalid_input_is_rejected(client):
    r = client.post("/predict", json={**VALID_TXN, "hour": 25})
    assert r.status_code == 422


def test_missing_required_field_is_rejected(client):
    txn = {k: v for k, v in VALID_TXN.items() if k != "amount"}
    r = client.post("/predict", json=txn)
    assert r.status_code == 422


def test_batch_scores_every_transaction(client):
    r = client.post("/predict/batch", json=[VALID_TXN, {**VALID_TXN, "amount": 10.0}])
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_predict_without_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "predictor", FraudPredictor(model_dir=str(tmp_path)))
    with TestClient(api.app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", json=VALID_TXN).status_code == 503


def test_batch_of_one(client):
    r = client.post("/predict/batch", json=[VALID_TXN])
    assert r.status_code == 200
    assert r.json()["count"] == 1


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/docs"


def test_health_reports_the_live_revision(client, monkeypatch):
    monkeypatch.setattr(api, "REVISION", "abc1234")
    assert client.get("/health").json()["revision"] == "abc1234"


def test_drift_endpoint_without_a_reference_returns_503(model_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "predictor", FraudPredictor(model_dir=str(model_dir)))
    monkeypatch.setattr(api, "REFERENCE_PATH", str(tmp_path / "missing.json"))
    with TestClient(api.app) as c:
        assert c.get("/monitoring/drift").status_code == 503
