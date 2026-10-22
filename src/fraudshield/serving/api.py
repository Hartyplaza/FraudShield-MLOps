"""
src/fraudshield/serving/api.py

FraudShield V2 — FastAPI serving endpoint.

Endpoints:
    GET  /health    — liveness check + model status
    POST /predict   — score a single transaction with SHAP explanation
    POST /predict/batch — score multiple transactions
    POST /score     — score one raw log line using each user's history

Run locally:
    uvicorn fraudshield.serving.api:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fraudshield.serving.predictor import FraudPredictor
from fraudshield.serving.schema import FraudResponse, HealthResponse, TransactionRequest
from fraudshield.serving.scorer import RawScorer, UnparseableLog

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION   = "0.1.0"
MODEL_DIR = os.getenv("MODEL_DIR", "data/processed/")
# Serve from the model registry instead, e.g. models:/fraudshield@champion
MODEL_URI = os.getenv("MODEL_URI")

app       = FastAPI(
    title       = "FraudShield V2",
    description = "Real-time fraud detection on transaction logs | Ofigwe Hart",
    version     = VERSION,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# Single predictor instance — loaded once on startup
predictor = FraudPredictor(model_dir=MODEL_DIR)
scorer: RawScorer | None = None
model_source: str | None = None


class RawLog(BaseModel):
    log: str


@app.on_event("startup")
def load_model():
    """Load model artefacts when the API starts."""
    global scorer, model_source
    scorer = None
    model_source = f"dir {predictor.model_dir}"
    if MODEL_URI:
        from pathlib import Path

        from fraudshield.registry.resolve import resolve_model_dir

        bundle, version = resolve_model_dir(MODEL_URI)
        predictor.model_dir = Path(bundle)
        model_source = f"{MODEL_URI} (version {version})"
        logger.info("Serving %s", model_source)
    try:
        predictor.load()
        logger.info("Model loaded successfully.")
    except FileNotFoundError as e:
        logger.warning("Model not loaded: %s", e)
        return
    try:
        scorer = RawScorer.from_model_dir(predictor, predictor.model_dir)
    except FileNotFoundError as e:
        logger.warning("Raw-log scoring disabled, feature state missing: %s", e)


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
        model_source = model_source,
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


@app.post("/score")
def score(body: RawLog):
    """
    Score one raw transaction log line, in any of the 7 supported formats.

    Features are computed from this user's history up to now, and the
    transaction is then added to that history.
    """
    if scorer is None:
        raise HTTPException(
            status_code=503,
            detail="Raw-log scoring unavailable: model or feature state not loaded."
        )
    try:
        return scorer.score(body.log)
    except UnparseableLog as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
