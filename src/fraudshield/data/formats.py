"""
src/fraudshield/data/formats.py

Render a structured transaction record into one of the 7 raw log formats
understood by fraudshield.ingestion.parser.
"""

FORMATS = (
    "pipe_colon",
    "arrow_bracket",
    "pipe_bar",
    "dash_equals",
    "usr_pipe",
    "space_separated",
    "dmy_triple_colon",
)

# Formats whose log line has no currency field
NO_CURRENCY = ("pipe_colon", "space_separated")


def _na(value):
    """Missing values are written as the literal string 'None'."""
    return "None" if value is None else value


def render(record: dict, fmt: str) -> str:
    """Render one transaction record as a raw log line in the given format."""
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format: {fmt}")
    if fmt not in NO_CURRENCY and record["currency"] is None:
        raise ValueError(f"{fmt} requires a currency")

    ts  = record["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    uid = record["user_id"]
    typ = record["txn_type"]
    amt = f"{record['amount']:.2f}"
    cur = record["currency"]
    loc = _na(record["location"])
    dev = _na(record["device"])

    if fmt == "pipe_colon":
        return f"{ts}::{uid}::{typ}::{amt}::{loc}::{dev}"

    if fmt == "arrow_bracket":
        return f"{ts} >> [{uid}] did {typ} - amt={cur}{amt} - {loc} // dev:{dev}"

    if fmt == "pipe_bar":
        return f"{ts} | user: {uid} | txn: {typ} of {cur}{amt} from {loc} | device: {dev}"

    if fmt == "dash_equals":
        return f"{ts} - user={uid} - action={typ} {cur}{amt} - ATM: {loc} - device={dev}"

    if fmt == "usr_pipe":
        return f"usr:{uid}|{typ}|{cur}{amt}|{loc}|{ts}|{dev}"

    if fmt == "space_separated":
        return f"{uid} {ts} {typ} {amt} {loc} {dev}"

    # dmy_triple_colon
    dmy_ts = record["timestamp"].strftime("%d/%m/%Y %H:%M:%S")
    return f"{dmy_ts} ::: {uid} *** {typ.upper()} ::: amt:{amt}{cur} @ {loc} <{dev}>"