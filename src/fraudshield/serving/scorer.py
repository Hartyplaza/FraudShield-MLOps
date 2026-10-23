"""
src/fraudshield/serving/scorer.py

Score raw transaction log lines in real time.

For each incoming line:
    parse -> features from the user's history so far -> model -> decision
and the transaction is then added to that user's history, so the next one
sees it. Feature computation is the same FeatureState code used in training.
"""

import logging
import threading
from pathlib import Path

import joblib
import pandas as pd

from fraudshield.features.online import CategoryEncoder, FeatureState
from fraudshield.ingestion.parser import parse_log_line
from fraudshield.serving.predictor import FraudPredictor

logger = logging.getLogger(__name__)


class UnparseableLog(ValueError):
    """Raised when a log line matches none of the known formats."""


class RawScorer:
    """Real-time scorer: raw log line in, fraud decision out."""

    def __init__(self, predictor: FraudPredictor, state: FeatureState):
        self.predictor = predictor
        self.state = state
        # FastAPI runs sync endpoints in a thread pool, so two requests for the
        # same user could otherwise update that user's history at the same time.
        self._lock = threading.Lock()

    @classmethod
    def from_model_dir(cls, predictor: FraudPredictor, model_dir: str | Path) -> "RawScorer":
        model_dir = Path(model_dir)
        state = joblib.load(model_dir / "feature_state.joblib")
        # The encoder saved as JSON is the source of truth for category codes.
        state.encoder = CategoryEncoder.load(model_dir / "encoders.json")
        logger.info("Feature state loaded for %d users", len(state.users))
        return cls(predictor, state)

    def score(self, raw_line: str) -> dict:
        return self.score_with_features(raw_line)[0]

    def score_with_features(self, raw_line: str) -> tuple[dict, dict]:
        """Score a line; also return the model features, for monitoring."""
        txn = parse_log_line(raw_line)
        if txn is None:
            raise UnparseableLog("Log line does not match any known format")

        with self._lock:
            features = self.state.process(txn)

        result = self.predictor.predict(pd.DataFrame([features]))
        prob = result["fraud_probability"][0]
        return {
            "user_id": txn["user_id"],
            "timestamp": txn["timestamp"],
            "txn_type": txn["txn_type"],
            "amount": txn["amount"],
            "fraud_probability": prob,
            "is_fraud": result["is_fraud"][0],
            "risk_level": self.predictor._risk_level(prob),
            "threshold_used": result["threshold_used"],
        }, features