# The pipeline itself is defined once, in dvc.yaml. These are shortcuts.
.PHONY: all test lint clean

all:
	uv run dvc repro

test:
	uv run pytest

lint:
	uv run ruff check src tests

clean:
	rm -rf data/raw/logs.csv data/interim data/processed models
