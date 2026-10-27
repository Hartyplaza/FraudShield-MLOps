"""
Fail if dvc.lock is behind the code or config.

A pull request that changes code or a parameter a stage depends on, without
rerunning that stage, leaves dvc.lock out of date: the committed reports no
longer describe the committed code. This happened twice before this check.

Only what a review controls is checked: code (src/, configs/) and parameters.
Data and models are build outputs: missing on a fresh checkout (CI) and
rebuilt with a new MLflow run id on every local training run, so they are
ignored.

    uv run python -m fraudshield.pipeline.lockcheck
"""

import json
import subprocess
import sys

REVIEWED = ("src/", "configs/")   # a nested path (file -> key) is a parameter


def stale(status: dict) -> list[str]:
    found = []

    def walk(stage: str, path: list[str], value) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(stage, [*path, k], v)
        elif value == "modified" and (len(path) > 1 or path[0].startswith(REVIEWED)):
            found.append(f"{stage}: {' -> '.join(path)} modified")

    for stage, items in status.items():
        for item in items:
            if isinstance(item, str):
                found.append(f"{stage}: {item}")
            else:
                for kind, changes in item.items():
                    if kind == "changed command":
                        found.append(f"{stage}: command changed")
                    elif kind == "changed deps":
                        walk(stage, [], changes)
    return found


def main() -> None:
    out = subprocess.run(["dvc", "status", "--json"], capture_output=True, text=True, check=True).stdout
    problems = stale(json.loads(out or "{}"))
    if problems:
        print("dvc.lock is out of date with the code or config:")
        print("\n".join(f"  {p}" for p in problems))
        print("Rerun the affected stages (uv run dvc repro <stage>) and commit dvc.lock and the reports.")
        sys.exit(1)
    print("dvc.lock is up to date with the code and config.")


if __name__ == "__main__":
    main()
