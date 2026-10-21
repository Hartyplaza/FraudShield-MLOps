"""
Stage 7: register. A model that passed the quality gate is logged as an MLflow
pyfunc model on its training run, registered as a new numbered version, and
given the alias "challenger". It does not replace the champion: promotion is a
separate, deliberate step (fraudshield.registry.promote).

    uv run python -m fraudshield.registry.register
"""

import json
import logging
from pathlib import Path

import mlflow

from fraudshield.pipeline.config import ensure_parent, stage_args
from fraudshield.pipeline.tracking import connect

logger = logging.getLogger(__name__)

PYFUNC_CODE = str(Path(__file__).with_name("pyfunc_model.py"))
CHALLENGER = "challenger"


def register_bundle(model_dir: str, run_id: str, name: str, tags: dict) -> int:
    """Log the bundle as a pyfunc model on `run_id`, register it, alias it challenger."""
    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(name="model", python_model=PYFUNC_CODE,
                                       artifacts={"bundle": model_dir})
    version = mlflow.register_model(info.model_uri, name, tags=tags)
    mlflow.MlflowClient().set_registered_model_alias(name, CHALLENGER, version.version)
    return int(version.version)


def main() -> None:
    cfg = stage_args("Register a model that passed the quality gate.")
    paths, name = cfg["paths"], cfg["registry"]["model_name"]

    gate = json.loads(Path(paths["gate"]).read_text(encoding="utf-8"))
    if not gate["passed"]:
        raise SystemExit("register: quality gate did not pass; nothing registered")
    seeds = json.loads(Path(paths["seeds"]).read_text(encoding="utf-8"))
    run_id = json.loads((Path(paths["model_dir"]) / "run.json").read_text(encoding="utf-8"))["run_id"]

    connect(cfg)
    tags = {
        "gate": f"passed {gate['checks']}/{gate['checks']}",
        "seeds.pr_auc_mean": str(seeds["pr_auc_mean"]),
        "seeds.pr_auc_std": str(seeds["pr_auc_std"]),
    }
    version = register_bundle(paths["model_dir"], run_id, name, tags)

    report = {"model": name, "version": version, "alias": CHALLENGER, "run_id": run_id}
    with open(ensure_parent(paths["registry"]), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Registered %s version %d (run %s) as '%s'", name, version, run_id, CHALLENGER)


if __name__ == "__main__":
    main()