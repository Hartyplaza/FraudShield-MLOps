"""
src/fraudshield/serving/api.py

FraudShield V2 — FastAPI serving endpoint.

Endpoints:
    GET  /health    — liveness check + model status
    POST /predict   — score a single transaction with SHAP explanation
    POST /predict/batch — score multiple transactions

Run locally:
    uvicorn fraudshield.serving.api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from fraudshield.serving.schema import TransactionRequest, FraudResponse, HealthResponse
from fraudshield.serving.predictor import FraudPredictor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION   = "0.1.0"
MODEL_DIR = os.getenv("MODEL_DIR", "data/processed/")

app       = FastAPI(
    title       = "FraudShield V2",
    description = "Real-time fraud detection on transaction logs | Ofigwe Hart",
    version     = VERSION,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# Single predictor instance — loaded once on startup
predictor = FraudPredictor(model_dir=MODEL_DIR)


@app.on_event("startup")
def load_model():
    """Load model artefacts when the API starts."""
    try:
        predictor.load()
        logger.info("Model loaded successfully.")
    except FileNotFoundError as e:
        logger.warning("Model not loaded: %s", e)


@app.get("/health", response_model=HealthResponse)
def health():
    """
    Liveness and readiness check.
    Returns model load status and current threshold.
    """
    return HealthResponse(
        status       = "ok",
        model_loaded = predictor._loaded,
        threshold    = predictor.threshold or 0.0,
        version      = VERSION,
    )


@app.post("/predict", response_model=FraudResponse)
def predict(txn: TransactionRequest):
    """
    Score a single transaction and return fraud probability + SHAP explanation.

    The response includes:
        - fraud_probability: model's confidence this is fraud (0-1)
        - is_fraud: True if probability >= threshold
        - risk_level: low / medium / high / critical
        - threshold_used: the tuned decision threshold
        - top_features: top 5 SHAP contributors driving this prediction
    """
    if not predictor._loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training pipeline first."
        )

    row    = pd.DataFrame([txn.model_dump()])
    result = predictor.predict_with_explanation(row)

    return FraudResponse(**result)


@app.post("/predict/batch")
def predict_batch(transactions: list[TransactionRequest]):
    """
    Score a batch of transactions. Returns probabilities without SHAP
    (SHAP is expensive per-row — use /predict for individual explanations).
    """
    if not predictor._loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training pipeline first."
        )

    if len(transactions) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 1000 transactions per request."
        )

    rows   = pd.DataFrame([t.model_dump() for t in transactions])
    result = predictor.predict(rows)

    return {
        "count"            : len(transactions),
        "fraud_detected"   : sum(result["is_fraud"]),
        "threshold_used"   : result["threshold_used"],
        "predictions"      : [
            {
                "fraud_probability": result["fraud_probability"][i],
                "is_fraud"         : bool(result["is_fraud"][i]),
                "risk_level"       : predictor._risk_level(
                    result["fraud_probability"][i]),
            }
            for i in range(len(transactions))
        ],
    }
