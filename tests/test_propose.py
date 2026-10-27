"""Retraining proposals: windows that follow the training data, config edits, the lock check."""

import shutil

import pandas as pd
import yaml

from fraudshield.pipeline.config import (
    load_config,
    normal_traffic,
    now_normal,
    training_end,
)
from fraudshield.pipeline.lockcheck import stale
from fraudshield.retraining.propose import apply, contract_changes, recent_period

CFG = load_config()
AMOUNTS = CFG["monitoring"]["drill"]["scenarios"]["amounts_x3"]


def with_period(cfg: dict, period: dict, normal: dict | None = None) -> dict:
    out = {**cfg, "retraining": {**cfg["retraining"], "periods": [period]}}
    if normal is not None:
        out["monitoring"] = {**cfg["monitoring"], "normal": normal}
    return out


def test_after_training_windows_follow_the_training_data():
    assert training_end(CFG) == "2025-07-31"                 # 60 days from 2025-06-01
    extended = with_period(CFG, {"start": "2025-07-31", "days": 30})
    assert training_end(extended) == "2025-08-30"
    assert training_end(extended, plus_days=14) == "2025-09-13"


def test_normal_traffic_and_accepted_scenarios():
    assert normal_traffic(CFG) == CFG["generator"]          # nothing accepted yet
    assert not now_normal(CFG, AMOUNTS)
    accepted = with_period(CFG, {"start": "2025-07-31", "days": 30}, normal=dict(AMOUNTS))
    assert normal_traffic(accepted)["amount_max"] == AMOUNTS["amount_max"]
    assert now_normal(accepted, AMOUNTS)
    assert not now_normal(accepted, {})                      # clean traffic is never "a scenario"


def test_recent_period_starts_after_training_and_covers_the_longest_window():
    p = recent_period(CFG, "amounts_x3")
    assert p["start"] == training_end(CFG)
    assert p["days"] >= 30                                   # txn_*_30d must settle
    assert p["amount_max"] == AMOUNTS["amount_max"]


def test_contract_change_only_for_the_amount_range():
    v = CFG["validation"]
    df = pd.DataFrame({"amount": [10.0, v["amount_range"][1] * 1.2]})
    changes, _ = contract_changes(CFG, df)
    assert changes["amount_range"][1] >= df["amount"].max()


def test_apply_edits_the_config_and_keeps_comments(tmp_path):
    path = tmp_path / "pipeline.yaml"
    shutil.copy("configs/pipeline.yaml", path)
    first = recent_period(CFG, "amounts_x3")
    done = apply(str(path), CFG, first, "amounts_x3", {"amount_range": [0.01, 15000]})
    text = path.read_text(encoding="utf-8")
    new = yaml.safe_load(text)
    assert new["retraining"]["periods"] == [first]
    assert new["monitoring"]["normal"] == AMOUNTS
    assert new["validation"]["amount_range"] == [0.01, 15000]
    assert "# Today's normal traffic" in text                  # comments survive
    assert any("DATA CONTRACT" in d for d in done)

    second = {**first, "seed": 101, "start": training_end(new)}
    apply(str(path), new, second, "none", {})
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["retraining"]["periods"] == [first, second]


def test_lockcheck_flags_code_and_params_not_missing_data():
    status = {
        "train": [{"changed deps": {"data/processed/features.csv": "deleted"},
                   "changed outs": {"models": "not in cache"}}],
        "gate": [{"changed deps": {"models": "deleted", "src/fraudshield/serving": "modified",
                                   "configs/pipeline.yaml": {"gates": "modified"}}}],
    }
    assert stale(status) == ["gate: src/fraudshield/serving modified",
                             "gate: configs/pipeline.yaml -> gates modified"]
    assert stale({"train": [{"changed deps": {"data/x.csv": "deleted"}}]}) == []
    assert stale({"register": [{"changed deps": {"models": "modified"}}]}) == []   # a local retrain
