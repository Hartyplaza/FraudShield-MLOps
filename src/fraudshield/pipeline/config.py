"""
src/fraudshield/pipeline/config.py

Single place that loads configs/pipeline.yaml for every pipeline stage.
"""

import argparse
import logging
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
