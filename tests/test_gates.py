"""Quality gate logic: metric limits and golden cases."""

import yaml

from fraudshield.evaluation.gates import check_golden, check_metrics
from fraudshield.ingestion.parser import parse_log_line
from fraudshield.pipeline.config import load_config

GATES = load_config()["gates"]
GOOD = {"pr_auc": 0.991, "recall": 0.984, "precision": 0.909, "fpr": 0.004}


def failed(results: list[dict]) -> list[str]:
    return [r["check"] for r in results if not r["passed"]]


def test_current_model_metrics_pass():
    assert failed(check_metrics(GOOD, GATES["metrics"])) == []


def test_m2_baseline_metrics_fail_on_the_right_checks():
    baseline = {"pr_auc": 0.9384, "recall": 0.918, "precision": 0.7273, "fpr": 0.0141}
    assert failed(check_metrics(baseline, GATES["metrics"])) == [
        "pr_auc >= 0.95", "precision >= 0.85", "fpr <= 0.01"]


def test_limits_are_inclusive():
    at_limit = {"pr_auc": 0.95, "recall": 0.90, "precision": 0.85, "fpr": 0.01}
    assert failed(check_metrics(at_limit, GATES["metrics"])) == []


class FakeScorer:
    """Stands in for RawScorer: flags every line containing 'cashout'."""

    def __init__(self):
        self.seen = []

    def score(self, line: str) -> dict:
        self.seen.append(line)
        fraud = "cashout" in line
        return {"is_fraud": fraud, "fraud_probability": 1.0 if fraud else 0.0}


def test_golden_cases_compare_decision_with_expectation():
    cases = [
        {"name": "a", "expect": "fraud", "line": "x cashout x"},
        {"name": "b", "expect": "legit", "line": "x purchase x"},
        {"name": "c", "expect": "fraud", "line": "x purchase x"},
    ]
    assert failed(check_golden(FakeScorer(), cases)) == ["golden: c -> fraud"]


def test_history_is_scored_before_the_case():
    scorer = FakeScorer()
    check_golden(scorer, [{"name": "a", "expect": "legit", "history": ["h1", "h2"], "line": "x"}])
    assert scorer.seen == ["h1", "h2", "x"]


def test_golden_case_file_is_well_formed():
    with open(GATES["golden_cases"], encoding="utf-8") as f:
        cases = yaml.safe_load(f)["cases"]
    assert {c["expect"] for c in cases} == {"fraud", "legit"}
    users = set()
    for case in cases:
        for line in [*case.get("history", []), case["line"]]:
            assert parse_log_line(line) is not None, line
        user = parse_log_line(case["line"])["user_id"]
        assert user not in users, f"golden cases must not share users: {user}"
        users.add(user)