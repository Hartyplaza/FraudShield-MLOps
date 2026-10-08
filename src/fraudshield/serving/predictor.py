"""
src/fraudshield/serving/predictor.py

Inference utilities — load a saved model and score transactions.
Used by the FastAPI app and for batch scoring.
"""

import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class FraudPredictor:
    """
    Wraps a trained LightGBM/XGBoost pipeline for inference.

    Handles:
        - Model and threshold loading from disk
        - Feature validation and ordering
        - Single-transaction and batch scoring
        - SHAP explanations per prediction
    """

    def __init__(self, model_dir: str = "data/processed/"):
        self.model_dir    = Path(model_dir)
        self.model        = None
        self.threshold    = None
        self.feature_cols = None
        self._loaded      = False

    def load(self) -> None:
        """Load model, threshold, and feature list from disk."""
        model_path     = self.model_dir / "fraudshield_model.joblib"
        threshold_path = self.model_dir / "best_threshold.txt"
        features_path  = self.model_dir / "feature_cols.json"

        if not model_path.exists():
            raise FileNotFoundError(
                "Model not found at {}. Run training first.".format(model_path))

        self.model        = joblib.load(model_path)
        self.threshold    = float(threshold_path.read_text().strip())
        self.feature_cols = json.loads(features_path.read_text())
        self._loaded      = True

        logger.info("Model loaded from %s | threshold=%.2f | features=%s",
                    model_path, self.threshold, len(self.feature_cols))

    def _validate(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all required feature columns are present and in the right order.
        Missing columns are filled with -1 (the NaN fill value used in training).
        """
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = -1
                logger.warning("Missing feature '%s' — filled with -1", col)
        return X[self.feature_cols].fillna(-1)

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Score a single transaction or batch of transactions.

        Args:
            X: DataFrame with feature columns

        Returns:
            dict with fraud_probability, is_fraud, and threshold_used
        """
        if not self._loaded:
            self.load()

        X_valid   = self._validate(X)
        probas    = self.model.predict_proba(X_valid)[:, 1]
        is_fraud  = (probas >= self.threshold).astype(int)

        if len(X_valid) == 1:
            return {
                "fraud_probability": round(float(probas[0]), 4),
                "is_fraud"         : bool(is_fraud[0]),
                "threshold_used"   : self.threshold,
            }

        return {
            "fraud_probability": [round(float(p), 4) for p in probas],
            "is_fraud"         : is_fraud.tolist(),
            "threshold_used"   : self.threshold,
        }

    def predict_with_explanation(self, X: pd.DataFrame) -> dict:
        """
        Score a single transaction and return SHAP feature contributions.

        Args:
            X: single-row DataFrame

        Returns:
            predict() result plus top_features dict
        """
        from fraudshield.explainability.shap_analysis import (
            explain_prediction, top_contributors
        )

        if not self._loaded:
            self.load()

        X_valid = self._validate(X)
        result  = self.predict(X_valid)

        shap_vals = explain_prediction(self.model, X_valid)
        result["top_features"] = top_contributors(shap_vals, n=5)
        result["risk_level"]   = self._risk_level(result["fraud_probability"])

        return result

    @staticmethod
    def _risk_level(prob: float) -> str:
        """Map fraud probability to human-readable risk level."""
        if prob < 0.30:  return "low"
        if prob < 0.60:  return "medium"
        if prob < 0.85:  return "high"
        return "critical"
