"""
src/fraudshield/training/train.py

Training pipeline:
    - RandomizedSearchCV with StratifiedKFold across 4 models
    - SMOTE inside CV folds (prevents leakage)
    - ADASYN on final refit (adaptive oversampling on full training set)
    - Threshold tuning on validation set
    - MLflow experiment tracking
    - Artefact saving
"""

import json
import logging
import time
import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.stats import randint, uniform

import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

from fraudshield.features.engineer import FEATURE_COLS
from fraudshield.evaluation.metrics import evaluate
from fraudshield.evaluation.threshold_tuner import tune_threshold

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
N_ITER      = 20
CV_FOLDS    = 5
SCORING     = "average_precision"


def _build_pipelines() -> dict:
    """
    Build four ImbPipelines — one per model.
    SMOTE is used inside CV to prevent leakage.
    eval_metric excluded from XGBoost to prevent NaN scores in RandomizedSearchCV.
    """
    lr_pipeline = ImbPipeline([
        ("smote",  SMOTE(random_state=RANDOM_SEED)),
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ])
    lr_params = {
        "model__C"      : uniform(0.001, 10),
        "model__penalty": ["l1", "l2"],
        "model__solver" : ["liblinear"],
    }

    rf_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("model", RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)),
    ])
    rf_params = {
        "model__n_estimators"     : randint(100, 500),
        "model__max_depth"        : [4, 6, 8, 10, None],
        "model__min_samples_split": randint(2, 20),
        "model__min_samples_leaf" : randint(1, 10),
        "model__max_features"     : ["sqrt", "log2"],
    }

    # eval_metric intentionally excluded — causes NaN in RandomizedSearchCV
    xgb_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("model", XGBClassifier(
            random_state=RANDOM_SEED, verbosity=0, n_jobs=1,
        )),
    ])
    xgb_params = {
        "model__n_estimators"    : randint(100, 400),
        "model__max_depth"       : randint(3, 7),
        "model__learning_rate"   : uniform(0.01, 0.14),
        "model__subsample"       : uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_weight": randint(1, 10),
        "model__gamma"           : uniform(0, 0.2),
    }

    lgbm_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("model", lgb.LGBMClassifier(
            random_state=RANDOM_SEED, verbosity=-1, n_jobs=-1,
        )),
    ])
    lgbm_params = {
        "model__n_estimators"     : randint(100, 600),
        "model__max_depth"        : randint(3, 10),
        "model__learning_rate"    : uniform(0.01, 0.3),
        "model__subsample"        : uniform(0.6, 0.4),
        "model__num_leaves"       : randint(20, 100),
        "model__min_child_samples": randint(5, 50),
    }

    return {
        "LogisticRegression": (lr_pipeline,   lr_params),
        "RandomForest"      : (rf_pipeline,   rf_params),
        "XGBoost"           : (xgb_pipeline,  xgb_params),
        "LightGBM"          : (lgbm_pipeline, lgbm_params),
    }


def _search_xgboost(pipeline, param_space, X_train, y_train, cv) -> tuple:
    """
    Manual random search for XGBoost — bypasses RandomizedSearchCV NaN issue
    with XGBoost 2.x internal threading.
    """
    best_score, best_params, best_model = -1.0, {}, None
    rng = np.random.RandomState(RANDOM_SEED)

    for _ in range(N_ITER):
        params = {
            k: (v.rvs(random_state=rng) if hasattr(v, "rvs") else rng.choice(v))
            for k, v in param_space.items()
        }
        fold_scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            pipe = ImbPipeline([
                ("smote", SMOTE(random_state=RANDOM_SEED)),
                ("model", XGBClassifier(
                    n_estimators     = int(params["model__n_estimators"]),
                    max_depth        = int(params["model__max_depth"]),
                    learning_rate    = float(params["model__learning_rate"]),
                    subsample        = float(params["model__subsample"]),
                    colsample_bytree = float(params["model__colsample_bytree"]),
                    min_child_weight = int(params["model__min_child_weight"]),
                    gamma            = float(params["model__gamma"]),
                    random_state     = RANDOM_SEED, verbosity=0, n_jobs=1,
                )),
            ])
            try:
                pipe.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                proba = pipe.predict_proba(X_train.iloc[val_idx])[:, 1]
                score = average_precision_score(y_train.iloc[val_idx], proba)
                if not np.isnan(score):
                    fold_scores.append(score)
            except Exception:
                break

        if len(fold_scores) == CV_FOLDS:
            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score, best_params = mean_score, params

    # Refit best params on full training set
    best_model = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("model", XGBClassifier(
            n_estimators     = int(best_params["model__n_estimators"]),
            max_depth        = int(best_params["model__max_depth"]),
            learning_rate    = float(best_params["model__learning_rate"]),
            subsample        = float(best_params["model__subsample"]),
            colsample_bytree = float(best_params["model__colsample_bytree"]),
            min_child_weight = int(best_params["model__min_child_weight"]),
            gamma            = float(best_params["model__gamma"]),
            random_state     = RANDOM_SEED, verbosity=0, n_jobs=1,
        )),
    ])
    best_model.fit(X_train, y_train)
    return best_score, best_params, best_model


def train(df: pd.DataFrame,
          label_col: str = "is_fraud",
          model_dir: str = "data/processed/",
          config_path: str = "configs/pipeline.yaml") -> dict:
    """
    Full training pipeline: search -> select -> refit with ADASYN -> tune threshold -> save.

    Args:
        df:          feature matrix with label column
        label_col:   name of the label column
        model_dir:   directory to save artefacts
        config_path: YAML config path (optional)

    Returns:
        dict of best model name, metrics, and artefact paths
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].fillna(-1)
    y = df[label_col]

    # Three-way split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20,
        stratify=y_trainval, random_state=RANDOM_SEED)

    logger.info("Train: %s | Val: %s | Test: %s", len(X_train), len(X_val), len(X_test))

    cv     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    models = _build_pipelines()
    results = {}

    for name, (pipeline, param_space) in models.items():
        logger.info("Searching: %s ...", name)
        start = time.time()

        if name == "XGBoost":
            score, params, fitted = _search_xgboost(
                pipeline, param_space, X_train, y_train, cv)
            results[name] = {"score": score, "params": params, "model": fitted}
        else:
            search = RandomizedSearchCV(
                estimator=pipeline, param_distributions=param_space,
                n_iter=N_ITER, scoring=SCORING, cv=cv, refit=True,
                n_jobs=-1, random_state=RANDOM_SEED, verbose=0, error_score=0,
            )
            search.fit(X_train, y_train)
            results[name] = {
                "score" : search.best_score_,
                "params": search.best_params_,
                "model" : search.best_estimator_,
            }

        elapsed = time.time() - start
        logger.info("  CV PR-AUC: %.4f | Time: %.1fs", results[name]["score"], elapsed)

    # Select best model
    best_name  = max(results, key=lambda k: results[k]["score"])
    best_entry = results[best_name]
    logger.info("Best model: %s (CV PR-AUC=%.4f)", best_name, best_entry["score"])

    # Refit best model with ADASYN on full training set
    best_steps    = list(best_entry["model"].steps)
    best_steps[0] = ("adasyn", ADASYN(random_state=RANDOM_SEED))
    best_model    = ImbPipeline(best_steps)
    if best_name != "XGBoost":
        best_model.set_params(**best_entry["params"])
    best_model.fit(X_train, y_train)

    # Tune threshold on validation set
    threshold = tune_threshold(best_model, X_val, y_val)

    # Final evaluation on test set
    metrics = evaluate(best_model, X_test, y_test, threshold=threshold)

    # Save artefacts
    joblib.dump(best_model, model_dir / "fraudshield_model.joblib")
    (model_dir / "best_threshold.txt").write_text(str(threshold))
    (model_dir / "feature_cols.json").write_text(
        json.dumps(feature_cols, indent=2))

    results_df = X_test.copy()
    results_df["is_fraud_actual"]    = y_test.values
    results_df["fraud_probability"]  = best_model.predict_proba(X_test)[:, 1]
    results_df["is_fraud_predicted"] = (
        results_df["fraud_probability"] >= threshold).astype(int)
    results_df.to_csv(model_dir / "model_results.csv", index=False)

    logger.info("Artefacts saved to %s", model_dir)

    return {
        "best_model"  : best_name,
        "metrics"     : metrics,
        "threshold"   : threshold,
        "model_dir"   : str(model_dir),
    }
