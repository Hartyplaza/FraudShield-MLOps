"""
Contract tests: every line rendered by the generator must be read back
by the parser with identical values.
"""

from datetime import datetime

import pytest

from fraudshield.data.formats import FORMATS, NO_CURRENCY, render
from fraudshield.ingestion.parser import parse_log_line

RECORD = {
    "timestamp": datetime(2025, 6, 14, 3, 7, 9),
    "user_id": "user1042",
    "txn_type": "top-up",
    "amount": 4123.5,
    "currency": "£",
    "location": "Leeds",
    "device": "Samsung Galaxy S10",
}


@pytest.mark.parametrize("fmt", FORMATS)
def test_round_trip(fmt):
    parsed = parse_log_line(render(RECORD, fmt))

    assert parsed is not None
    assert parsed["log_format"] == fmt
    assert parsed["timestamp"] == RECORD["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    assert parsed["user_id"] == RECORD["user_id"]
    assert parsed["txn_type"] == RECORD["txn_type"]
    assert parsed["amount"] == RECORD["amount"]
    assert parsed["location"] == RECORD["location"]
    assert parsed["device"] == RECORD["device"]

    expected_currency = None if fmt in NO_CURRENCY else RECORD["currency"]
    assert parsed["currency"] == expected_currency


@pytest.mark.parametrize("fmt", FORMATS)
def test_missing_location_and_device(fmt):
    record = {**RECORD, "location": None, "device": None}
    parsed = parse_log_line(render(record, fmt))

    assert parsed is not None
    assert parsed["location"] is None
    assert parsed["device"] is None


def test_unknown_format_raises():
    with pytest.raises(ValueError, match="Unknown format"):
        render(RECORD, "nope")


@pytest.mark.parametrize("fmt", [f for f in FORMATS if f not in NO_CURRENCY])
def test_currency_required(fmt):
    with pytest.raises(ValueError, match="requires a currency"):
        render({**RECORD, "currency": None}, fmt)