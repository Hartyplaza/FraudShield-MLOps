"""Seed robustness: the recipe is refitted on a freshly generated dataset."""

from lightgbm import LGBMClassifier

from fraudshield.features.engineer import FEATURE_COLS
from fraudshield.pipeline.config import load_config
from fraudshield.pipeline.seeds import run_seed


def test_run_seed_fits_and_evaluates_on_new_data():
    cfg = load_config()
    cfg["generator"]["n_rows"] = 3000
    recipe = LGBMClassifier(n_estimators=50, verbose=-1, random_state=0)
    result = run_seed(cfg, seed=3, recipe=recipe, feature_cols=FEATURE_COLS)
    assert result["seed"] == 3 and result["test_frauds"] > 0
    assert 0 < result["pr_auc"] <= 1 and 0 < result["threshold"] < 1
    assert result["tp"] + result["fn"] == result["test_frauds"]