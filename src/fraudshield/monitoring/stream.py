"""
src/fraudshield/monitoring/stream.py

Score a generated stream the way the API does: each line is parsed, its
features come from the user's history so far (starting from the bundle's
saved state), and the model scores it. Used to build the drift reference and
to run the drift drills.
"""

import copy
from dataclasses import dataclass

import pandas as pd

from fraudshield.ingestion.parser import parse_log_line
from fraudshield.serving.predictor import FraudPredictor
from fraudshield.serving.scorer import RawScorer
from fraudshield.serving.stream_demo import build_stream


@dataclass
class ScoredStream:
    features: pd.DataFrame   # model features, before NaN filling
    scores: pd.Series
    labels: pd.Series        # ground truth from the labelling rules
    unparseable: int
    threshold: float


def score_stream(model_dir: str, generator_cfg: dict) -> ScoredStream:
    lines, rejected, labels = build_stream(generator_cfg)
    predictor = FraudPredictor(model_dir)
    predictor.load()
    scorer = RawScorer.from_model_dir(predictor, model_dir)
    state = copy.deepcopy(scorer.state)   # every stream starts from the same history

    rows = [state.process(parse_log_line(line)) for line in lines]
    features = pd.DataFrame(rows)[predictor.feature_cols]
    # The model is stateless, so scoring the rows together equals scoring them one by one
    scores = pd.Series(predictor.predict(features.copy())["fraud_probability"], name="score")
    return ScoredStream(features, scores, labels.reset_index(drop=True), len(rejected), predictor.threshold)
