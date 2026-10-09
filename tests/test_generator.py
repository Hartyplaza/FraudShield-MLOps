"""
Tests for the synthetic log generator.

The end-to-end test runs generated data through the real parser, cleaner
and labeller, so it fails if any stage stops agreeing with the generator.
"""

import logging

import pandas as pd
import pytest
import yaml

from fraudshield.data.formats import FORMATS
from fraudshield.data.generate import MALFORMED, generate, write_csv
from fraudshield.ingestion.cleaner import clean
from fraudshield.ingestion.parser import parse_logs
from fraudshield.labelling.label_simulator import simulate_labels


@pytest.fixture(scope="module")
def cfg():
    with open("configs/pipeline.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["generator"]


@pytest.fixture(scope="module")
def rows(cfg):
    return generate(cfg)


@pytest.fixture(scope="module")
def parsed(rows):
    logging.disable(logging.INFO)
    return parse_logs(pd.DataFrame({"raw_log": rows}))


def test_row_count(cfg, rows):
    assert len(rows) == cfg["n_rows"]


def test_same_seed_same_file(cfg, rows, tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_csv(rows, a)
    write_csv(generate(cfg), b)
    assert a.read_bytes() == b.read_bytes()


def test_different_seed_different_data(cfg, rows):
    assert generate({**cfg, "seed": cfg["seed"] + 1}) != rows


def test_noise_rates(cfg, rows):
    n = len(rows)
    null_rate = sum(r is None for r in rows) / n
    malformed_rate = sum(r == MALFORMED for r in rows) / n
    assert abs(null_rate - cfg["null_rate"]) < 0.02
    assert abs(malformed_rate - cfg["malformed_rate"]) < 0.02


def test_every_valid_row_parses(rows, parsed):
    valid = sum(r is not None and r != MALFORMED for r in rows)
    assert len(parsed) == valid


def test_all_formats_present(parsed):
    assert set(parsed["log_format"]) == set(FORMATS)


def test_locations_are_single_tokens(cfg):
    # Five parser patterns capture location with \S+, so spaces would break parsing
    assert all(" " not in loc for loc in cfg["locations"])


def test_fraud_prevalence_matches_assessment(parsed):
    labelled = simulate_labels(clean(parsed))
    rate = labelled["is_fraud"].mean()
    assert 0.030 <= rate <= 0.045, f"fraud rate {rate:.3%} outside V2 band"