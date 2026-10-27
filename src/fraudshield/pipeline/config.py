"""
src/fraudshield/pipeline/config.py

Single place that loads configs/pipeline.yaml for every pipeline stage.
"""

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import yaml

DEFAULT_CONFIG = "configs/pipeline.yaml"


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def stage_args(description: str) -> dict:
    """Parse the common --config flag, set up logging, and return the config."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    return load_config(args.config)


def ensure_parent(path: str) -> Path:
    """Create the parent folder of an output file and return it as a Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def training_periods(cfg: dict) -> list[dict]:
    """Generator settings for every period of training data: the original one, then
    each labelled period added by retraining (configs/pipeline.yaml -> retraining.periods)."""
    base = cfg["generator"]
    return [base] + [{**base, **extra} for extra in cfg.get("retraining", {}).get("periods", [])]


def training_end(cfg: dict, plus_days: int = 0) -> str:
    """The day after the training data ends (+ plus_days), as YYYY-MM-DD.

    Everything that means "traffic after training" (the drift reference, the
    drills, the live demo) starts here, so it moves when retraining adds data.
    """
    end = max(date.fromisoformat(p["start"]) + timedelta(days=p["days"]) for p in training_periods(cfg))
    return (end + timedelta(days=plus_days)).isoformat()


def normal_traffic(cfg: dict) -> dict:
    """Generator settings for today's normal traffic: the original generator plus
    monitoring.normal, the changes accepted as the new normal by a retraining."""
    return {**cfg["generator"], **cfg["monitoring"].get("normal", {})}


def now_normal(cfg: dict, change: dict) -> bool:
    """Whether a drift scenario's changes are already part of normal traffic."""
    normal = cfg["monitoring"].get("normal", {})
    return bool(change) and all(normal.get(k) == v for k, v in change.items())
