"""
Stage 3: labelled transactions -> feature matrix.

Features are built by replaying transactions in time order through the same
FeatureState the API uses, so training and serving compute features identically
and no feature can see the future.

Outputs:
    paths.features       feature matrix (original columns + 27 features)
    paths.encoders       category -> code mapping learned from this data
    paths.feature_state  every user's history at the end of the data, which
                         the API continues from

    uv run python -m fraudshield.pipeline.features
"""

import logging

import joblib
import pandas as pd

from fraudshield.features.online import CategoryEncoder, replay
from fraudshield.pipeline.config import ensure_parent, stage_args
from fraudshield.validation.schemas import features_schema, validate

logger = logging.getLogger(__name__)


def main() -> None:
    cfg = stage_args("Build the 27-feature matrix by replaying transactions.")
    paths = cfg["paths"]

    df = pd.read_csv(paths["labelled"], parse_dates=["timestamp"])
    encoder = CategoryEncoder().fit(df)
    features, state = replay(df, encoder)
    validate(features, features_schema(cfg["validation"]), "features")

    out = pd.concat([df, features], axis=1)
    out.to_csv(ensure_parent(paths["features"]), index=False)
    encoder.save(ensure_parent(paths["encoders"]))
    joblib.dump(state, ensure_parent(paths["feature_state"]))

    logger.info("Feature matrix %s | state for %d users", out.shape, len(state.users))


if __name__ == "__main__":
    main()