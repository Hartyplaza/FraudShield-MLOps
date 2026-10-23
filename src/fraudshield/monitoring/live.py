"""
src/fraudshield/monitoring/live.py

Drift monitoring inside the running API. Keeps the features and score of the
last `window` transactions it scored, plus whether each recent request could be
parsed, and compares them with the reference profile on request.

Held in memory: the window restarts empty whenever the service restarts.
"""

import threading
from collections import deque

import pandas as pd

from fraudshield.monitoring.drift import drift_report, load_reference


class LiveMonitor:
    def __init__(self, reference: dict):
        self.reference = reference
        self.limits = reference["limits"]
        window = self.limits["window"]
        self._rows: deque[dict] = deque(maxlen=window)
        self._scores: deque[float] = deque(maxlen=window)
        self._parsed: deque[bool] = deque(maxlen=window)   # True: scored, False: unparseable
        self._lock = threading.Lock()
        self.total = 0

    @classmethod
    def from_file(cls, path: str) -> "LiveMonitor":
        return cls(load_reference(path))

    def record(self, features: dict, score: float) -> None:
        with self._lock:
            self._rows.append(features)
            self._scores.append(score)
            self._parsed.append(True)
            self.total += 1

    def record_unparseable(self) -> None:
        with self._lock:
            self._parsed.append(False)
            self.total += 1

    def report(self) -> dict:
        with self._lock:
            rows, scores, parsed = list(self._rows), list(self._scores), list(self._parsed)
            total = self.total
        cols = list(self.reference["features"])
        features = pd.DataFrame(rows, columns=cols)
        # Unparseable share over the recent requests, expressed against the scored window
        bad = parsed.count(False)
        unparseable = round(bad * len(rows) / parsed.count(True)) if parsed.count(True) else bad
        report = drift_report(self.reference, features, pd.Series(scores, dtype=float),
                              self.limits, unparseable=unparseable)
        return {"window": self.limits["window"], "requests_since_start": total, **report}
