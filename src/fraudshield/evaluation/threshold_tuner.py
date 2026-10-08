"""
src/fraudshield/evaluation/threshold_tuner.py

Threshold tuning for imbalanced fraud classification.

The default 0.5 threshold is wrong for imbalanced data — with 3.4% fraud,
a model with well-separated probabilities needs a much lower or higher
threshold to maximise F1. We sweep the full range and find the peak.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def tune_threshold(model,
                   X_val: pd.DataFrame,
                   y_val: pd.Series,
                   min_t: float = 0.01,
                   max_t: float = 0.95,
                   step: float  = 0.01) -> float:
    """
    Sweep thresholds from min_t to max_t and return the one that
    maximises F1 score on the validation set.

    Uses the validation set — never the test set — to avoid
    threshold optimism bias on the final evaluation.

    Args:
        model:  trained pipeline with predict_proba
        X_val:  validation features
        y_val:  validation labels
        min_t:  minimum threshold to try
        max_t:  maximum threshold to try
        step:   step size between thresholds

    Returns:
        best_threshold as float
    """
    y_proba = model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(min_t, max_t + step, step)
    best_t, best_f1 = 0.5, 0.0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f = f1_score(y_val, y_pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t  = round(float(t), 2)

    logger.info("Optimal threshold: %.2f  (F1 = %.4f)", best_t, best_f1)
    return best_t


def threshold_curve(model,
                    X_val: pd.DataFrame,
                    y_val: pd.Series,
                    min_t: float = 0.01,
                    max_t: float = 0.95,
                    step: float  = 0.01) -> pd.DataFrame:
    """
    Return a DataFrame of precision, recall, and F1 at each threshold.
    Useful for plotting the threshold curve in notebooks.

    Returns:
        DataFrame with columns: threshold, f1, precision, recall
    """
    from sklearn.metrics import precision_score, recall_score

    y_proba    = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(min_t, max_t + step, step)
    rows       = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold" : round(float(t), 2),
            "f1"        : f1_score(y_val, y_pred, zero_division=0),
            "precision" : precision_score(y_val, y_pred, zero_division=0),
            "recall"    : recall_score(y_val, y_pred, zero_division=0),
        })

    return pd.DataFrame(rows)
