"""
src/fraudshield/ingestion/parser.py

Parses raw multi-format transaction log lines into structured records.
Supports 7 distinct log schemas via regex pattern dispatch.
"""

import re
import logging
import pandas as pd
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Each pattern uses named groups so we always know which field is which,
# regardless of the order they appear in a given format.
# ---------------------------------------------------------------------------
PATTERNS = {

    # Format 1 — pipe_colon
    # 2025-07-05 19:18:10::user1069::withdrawal::2995.12::London::iPhone 13
    "pipe_colon": re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r"::(?P<user_id>user\d+)"
        r"::(?P<txn_type>\w[\w-]*)"
        r"::(?P<amount>[\d.]+)"
        r"::(?P<location>[^:]+)"
        r"::(?P<device>.+)"
    ),

    # Format 2 — arrow_bracket
    # 2025-07-20 05:38:14 >> [user1034] did top-up - amt=€2191.06 - None // dev:iPhone 13
    "arrow_bracket": re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r"\s*>>\s*\[(?P<user_id>user\d+)\]\s*did\s+(?P<txn_type>[\w-]+)"
        r"\s*-\s*amt=(?P<currency>[€£\$])(?P<amount>[\d.]+)"
        r"\s*-\s*(?P<location>\S+)\s*//\s*dev:(?P<device>.+)"
    ),

    # Format 3 — pipe_bar
    # 2025-07-31 06:50:50 | user: user1071 | txn: cashout of $1772.13 from None | device: Nokia 3310
    "pipe_bar": re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r"\s*\|\s*user:\s*(?P<user_id>user\d+)"
        r"\s*\|\s*txn:\s*(?P<txn_type>[\w-]+)\s+of\s+(?P<currency>[€£\$])(?P<amount>[\d.]+)"
        r"\s+from\s+(?P<location>\S+)"
        r"\s*\|\s*device:\s*(?P<device>.+)"
    ),

    # Format 4 — dash_equals
    # 2025-07-23 15:57:12 - user=user1098 - action=purchase €2019.47 - ATM: Glasgow - device=iPhone 13
    "dash_equals": re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r"\s*-\s*user=(?P<user_id>user\d+)"
        r"\s*-\s*action=(?P<txn_type>[\w-]+)\s+(?P<currency>[€£\$])(?P<amount>[\d.]+)"
        r"\s*-\s*ATM:\s*(?P<location>\S+)"
        r"\s*-\s*device=(?P<device>.+)"
    ),

    # Format 5 — usr_pipe
    # usr:user1076|cashout|€4821.85|Glasgow|2025-07-15 12:56:05|Pixel 6
    "usr_pipe": re.compile(
        r"usr:(?P<user_id>user\d+)"
        r"\|(?P<txn_type>[\w-]+)"
        r"\|(?P<currency>[€£\$])(?P<amount>[\d.]+)"
        r"\|(?P<location>[^|]+)"
        r"\|(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
        r"\|(?P<device>.+)"
    ),

    # Format 6 — space_separated
    # user1093 2025-07-05 14:11:06 withdrawal 4926.56 None Huawei P30
    "space_separated": re.compile(
        r"(?P<user_id>user\d+)\s+"
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
        r"(?P<txn_type>[\w-]+)\s+"
        r"(?P<amount>[\d.]+)\s+"
        r"(?P<location>\S+)\s+"
        r"(?P<device>.+)"
    ),

    # Format 7 — dmy_triple_colon
    # 24/07/2025 22:47:06 ::: user1080 *** PURCHASE ::: amt:951.85$ @ Liverpool <Xiaomi Mi 11>
    "dmy_triple_colon": re.compile(
        r"(?P<day>\d{2})/(?P<month>\d{2})/(?P<year>\d{4})\s+(?P<time>\d{2}:\d{2}:\d{2})"
        r"\s+:::\s+(?P<user_id>user\d+)"
        r"\s+\*\*\*\s+(?P<txn_type>[A-Z\-]+)"
        r"\s+:::\s+amt:(?P<amount>[\d.]+)(?P<currency>[€£\$])"
        r"\s+@\s+(?P<location>\S+)"
        r"\s+<(?P<device>[^>]+)>"
    ),
}

# Fields that should be treated as missing when they contain these strings
NONE_PLACEHOLDERS = {"none", "null", "n/a", "na", "-", ""}


def _normalise_timestamp(match: re.Match, fmt_name: str) -> Optional[str]:
    """
    All formats except dmy_triple_colon store timestamp as a single group.
    dmy_triple_colon splits day/month/year/time into 4 groups — we reassemble.
    """
    if fmt_name == "dmy_triple_colon":
        return (
            f"{match.group('year')}-{match.group('month')}-{match.group('day')}"
            f" {match.group('time')}"
        )
    return match.group("timestamp")


def _clean_field(value: Optional[str]) -> Optional[str]:
    """Return None for any known missing-value placeholder, else strip whitespace."""
    if value is None:
        return None
    stripped = value.strip()
    if stripped.lower() in NONE_PLACEHOLDERS:
        return None
    return stripped


def parse_log_line(raw) -> Optional[dict]:
    """
    Try every pattern against a single raw log string.
    Returns a dict of fields on success, None if the line cannot be parsed.
    """
    # --- Guard: skip nulls and known bad markers ---
    if pd.isna(raw) or str(raw).strip().upper() == "MALFORMED_LOG" or str(raw).strip() == "":
        return None

    raw = str(raw)

    for fmt_name, pattern in PATTERNS.items():
        m = pattern.search(raw)
        if not m:
            continue                          # try the next pattern

        gd = m.groupdict()

        # --- Amount: must be a valid float ---
        try:
            amount = float(gd.get("amount", ""))
        except (ValueError, TypeError):
            continue                          # malformed amount → skip

        # --- Timestamp normalisation ---
        timestamp = _normalise_timestamp(m, fmt_name)

        return {
            "timestamp":  timestamp,
            "user_id":    _clean_field(gd.get("user_id")),
            "txn_type":   _clean_field(gd.get("txn_type", "").lower()),
            "amount":     amount,
            "currency":   _clean_field(gd.get("currency")),
            "location":   _clean_field(gd.get("location")),
            "device":     _clean_field(gd.get("device")),
            "log_format": fmt_name,
        }

    # No pattern matched
    logger.debug("Unmatched line: %s", raw[:80])
    return None


def parse_logs(df: pd.DataFrame, col: str = "raw_log") -> pd.DataFrame:
    """
    Apply parse_log_line to every row in df[col].
    Drops unparseable rows and converts timestamp to datetime.

    Returns a clean structured DataFrame.
    """
    total = len(df)
    logger.info("Parsing %s raw log rows...", f"{total:,}")

    parsed  = df[col].apply(parse_log_line)
    records = [r for r in parsed if r is not None]

    result = pd.DataFrame(records)
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")

    dropped = total - len(result)
    logger.info(
        "Parsed: %s rows kept | %s dropped (%.1f%% noise)",
        f"{len(result):,}",
        f"{dropped:,}",
        dropped / total * 100,
    )
    return result
