"""
src/fraudshield/registry/resolve.py

Turn a registry address such as models:/fraudshield@champion into a local model
bundle directory, so the API can serve whichever version the alias points to.
"""

import os
from pathlib import Path


def resolve_model_dir(model_uri: str) -> tuple[Path, str]:
    """Download the registered model and return (bundle directory, version)."""
    import mlflow  # only needed when serving from the registry

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    root = Path(mlflow.artifacts.download_artifacts(model_uri))
    bundle = next(p.parent for p in root.glob("artifacts/*/fraudshield_model.joblib"))

    name, _, ref = model_uri.removeprefix("models:/").partition("@")
    client = mlflow.MlflowClient()
    version = (client.get_model_version_by_alias(name, ref).version if ref
               else model_uri.rsplit("/", 1)[-1])
    return bundle, str(version)