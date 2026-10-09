"""
src/fraudshield/explainability/shap_analysis.py

SHAP-based explainability for fraud predictions.

Two functions:
    global_importance  — beeswarm + bar plots saved to reports/figures/
    explain_prediction — per-transaction SHAP values for the API response
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_classifier(model):
    """
    Extract the base classifier from inside an imblearn or sklearn Pipeline.
    Handles both 'model' and 'classifier' step names.
    """
    if hasattr(model, "named_steps"):
        for key in ["model", "classifier"]:
            if key in model.named_steps:
                return model.named_steps[key]
    return model


def global_importance(model,
                      X: pd.DataFrame,
                      out_dir: str = "reports/figures") -> None:
    """
    Generate and save SHAP global importance plots.

    Saves two files:
        shap_summary.png  — beeswarm plot (direction + magnitude per sample)
        shap_bar.png      — mean absolute SHAP per feature (ranking)

    Args:
        model:   trained pipeline
        X:       feature matrix (test or validation set)
        out_dir: directory to save plots
    """
    import shap
    import matplotlib.pyplot as plt

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    classifier  = _get_classifier(model)
    explainer   = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X)

    # LightGBM returns a list for binary classification — take class 1
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Beeswarm plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X, show=False)
    plt.tight_layout()
    beeswarm_path = "{}/shap_summary.png".format(out_dir)
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()
    logger.info("SHAP beeswarm saved -> %s", beeswarm_path)

    # Bar plot
    plt.figure(figsize=(9, 8))
    shap.summary_plot(sv, X, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = "{}/shap_bar.png".format(out_dir)
    plt.savefig(bar_path, dpi=150)
    plt.close()
    logger.info("SHAP bar plot saved -> %s", bar_path)


def explain_prediction(model,
                       X_row: pd.DataFrame) -> dict:
    """
    Return SHAP values for a single transaction.
    Used by the FastAPI /predict endpoint to explain each decision.

    Args:
        model: trained pipeline
        X_row: single-row DataFrame with the same columns as the training data

    Returns:
        dict mapping feature name -> SHAP value (positive = pushes toward fraud)
    """
    import shap

    classifier  = _get_classifier(model)
    explainer   = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_row)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    return {
        col: round(float(val), 5)
        for col, val in zip(X_row.columns, sv)
    }


def top_contributors(shap_dict: dict, n: int = 5) -> dict:
    """
    Return the top N features by absolute SHAP value.
    Used by the API to show the most influential factors per prediction.

    Args:
        shap_dict: output of explain_prediction()
        n:         number of top features to return

    Returns:
        dict of top N features sorted by absolute impact descending
    """
    sorted_items = sorted(
        shap_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return dict(sorted_items[:n])
