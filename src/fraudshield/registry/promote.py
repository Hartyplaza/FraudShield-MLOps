"""
Promotion: decide whether the challenger replaces the champion.

The rule (configs/pipeline.yaml -> promotion):
    Both models are refitted and evaluated on the same seed datasets (a paired
    comparison). The challenger is promoted if its mean PR-AUC is not lower
    than the champion's by more than max_pr_auc_drop. With no champion yet,
    the challenger is promoted outright.

Promotion only moves the "champion" alias; no files are copied. Every move is
appended to reports/promotions.jsonl (committed to Git), and the previous
champion is recorded on the new one, so rolling back is one command:

    uv run python -m fraudshield.registry.promote               # challenger vs champion
    uv run python -m fraudshield.registry.promote --version 3 --force   # pin a version
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from fraudshield.pipeline.config import ensure_parent, load_config
from fraudshield.pipeline.seeds import run_seed
from fraudshield.pipeline.tracking import connect
from fraudshield.registry.register import CHALLENGER

logger = logging.getLogger(__name__)

CHAMPION = "champion"


def load_recipe(name: str, version: int) -> tuple:
    """The fitted model and feature list of a registered version."""
    root = Path(mlflow.artifacts.download_artifacts(f"models:/{name}/{version}"))
    bundle = next(p.parent for p in root.glob("artifacts/*/fraudshield_model.joblib"))
    feature_cols = json.loads((bundle / "feature_cols.json").read_text(encoding="utf-8"))
    return joblib.load(bundle / "fraudshield_model.joblib"), feature_cols


def compare(cfg: dict, name: str, challenger: int, champion: int) -> pd.DataFrame:
    """PR-AUC of both versions on each seed dataset, side by side."""
    rows = []
    recipes = {"challenger": load_recipe(name, challenger), "champion": load_recipe(name, champion)}
    for seed in cfg["seeds"]["seeds"]:
        row = {"seed": seed}
        for role, (model, cols) in recipes.items():
            row[role] = run_seed(cfg, seed, model, cols)["pr_auc"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table["diff"] = table["challenger"] - table["champion"]
    return table


def decide(table: pd.DataFrame, max_drop: float) -> tuple[bool, str]:
    """Promote unless the challenger is worse than the champion by more than max_drop."""
    mean_diff = table["diff"].mean()
    wins = int((table["diff"] > 0).sum())
    summary = (f"mean PR-AUC change {mean_diff:+.4f} (challenger better on {wins} of "
               f"{len(table)} seeds; allowed drop {max_drop})")
    return mean_diff >= -max_drop, summary


def set_champion(client, name: str, version: int, reason: str, log_path: str | None = None) -> None:
    """Move the champion alias. Aliases keep no history, so each move is also
    appended to a log file that is committed to Git."""
    previous = current_version(client, name, CHAMPION)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    client.set_registered_model_alias(name, CHAMPION, version)
    client.set_model_version_tag(name, version, "promoted_at", now)
    client.set_model_version_tag(name, version, "promotion_reason", reason)
    client.set_model_version_tag(name, version, "previous_champion", str(previous or "none"))
    if log_path:
        entry = {"time": now, "model": name, "from": previous, "to": version, "reason": reason}
        with open(ensure_parent(log_path), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    logger.info("champion: version %s -> version %d", previous or "none", version)


def current_version(client, name: str, alias: str) -> int | None:
    try:
        return int(client.get_model_version_by_alias(name, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote the challenger if it is good enough.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--version", type=int, help="version to promote (default: the challenger)")
    parser.add_argument("--force", action="store_true", help="skip the comparison (rollback)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    cfg = load_config(args.config)
    connect(cfg)
    client = mlflow.MlflowClient()
    name = cfg["registry"]["model_name"]
    log = cfg["paths"]["promotions"]

    candidate = args.version or current_version(client, name, CHALLENGER)
    champion = current_version(client, name, CHAMPION)
    if candidate is None:
        raise SystemExit("No challenger registered; run the pipeline first")
    if candidate == champion:
        logger.info("Version %d is already the champion; nothing to do", candidate)
        return

    if args.force:
        set_champion(client, name, candidate, "forced (manual promotion or rollback)", log)
        return
    if champion is None:
        set_champion(client, name, candidate, "first champion", log)
        return

    logging.getLogger("fraudshield").setLevel(logging.WARNING)
    table = compare(cfg, name, candidate, champion)
    logging.getLogger("fraudshield").setLevel(logging.INFO)
    promote, summary = decide(table, cfg["promotion"]["max_pr_auc_drop"])

    logger.info("Version %d (challenger) vs version %d (champion), PR-AUC per seed:\n%s",
                candidate, champion, table.round(4).to_string(index=False))
    if promote:
        set_champion(client, name, candidate, summary, log)
    else:
        client.set_model_version_tag(name, candidate, "rejected", summary)
        raise SystemExit(f"NOT promoted: {summary}. Champion stays version {champion}.")


if __name__ == "__main__":
    main()