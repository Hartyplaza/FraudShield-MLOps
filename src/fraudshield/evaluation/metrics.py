"""
src/fraudshield/evaluation/metrics.py

Evaluation metrics for fraud detection.
Primary metric: PR-AUC (correct for imbalanced classification).
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series,
             threshold: float = 0.5) -> dict:
    """
    Compute all evaluation metrics for a trained model on the test set.

    Args:
        model:     trained sklearn/LightGBM/XGBoost pipeline
        X_test:    feature matrix
        y_test:    true labels
        threshold: decision threshold for binary classification

    Returns:
        dict of metric name -> value
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "pr_auc"    : round(float(average_precision_score(y_test, y_proba)), 4),
        "roc_auc"   : round(float(roc_auc_score(y_test, y_proba)), 4),
        "f1"        : round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "precision" : round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall"    : round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "fpr"       : round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4),
        "threshold" : threshold,
        "tp"        : int(tp),
        "fp"        : int(fp),
        "fn"        : int(fn),
        "tn"        : int(tn),
    }

    logger.info("=" * 45)
    logger.info("  PR-AUC    : %s  (primary)", metrics["pr_auc"])
    logger.info("  ROC-AUC   : %s", metrics["roc_auc"])
    logger.info("  F1        : %s", metrics["f1"])
    logger.info("  Precision : %s", metrics["precision"])
    logger.info("  Recall    : %s", metrics["recall"])
    logger.info("  FPR       : %s", metrics["fpr"])
    logger.info("  Threshold : %s", threshold)
    logger.info("  TP/FP/FN/TN: %s/%s/%s/%s", tp, fp, fn, tn)
    logger.info("=" * 45)

    return metrics
