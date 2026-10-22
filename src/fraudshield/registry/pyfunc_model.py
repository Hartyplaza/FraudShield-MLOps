"""
src/fraudshield/registry/pyfunc_model.py

The model bundle wrapped as an MLflow pyfunc model, so it can be registered,
versioned and loaded by name: mlflow.pyfunc.load_model("models:/fraudshield@champion").

Logged as "model from code": MLflow stores this file, not a pickled object, and
runs it when the model is loaded. Input: a DataFrame with a raw_log column.
Output: one row per line with the same fields as POST /score.
"""

import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel

from fraudshield.serving.predictor import FraudPredictor
from fraudshield.serving.scorer import RawScorer


class FraudShieldModel(PythonModel):
    def load_context(self, context) -> None:
        bundle = context.artifacts["bundle"]
        predictor = FraudPredictor(bundle)
        predictor.load()
        self.scorer = RawScorer.from_model_dir(predictor, bundle)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return pd.DataFrame([self.scorer.score(line) for line in model_input["raw_log"]])


mlflow.models.set_model(FraudShieldModel())