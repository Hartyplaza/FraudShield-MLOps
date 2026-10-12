CONFIG ?= configs/pipeline.yaml
PY     := uv run python -m

RAW      := data/raw/logs.csv
CLEAN    := data/interim/clean.csv
LABELLED := data/interim/labelled.csv
FEATURES := data/processed/features.csv
METRICS  := reports/metrics.json

PKG := src/fraudshield

.PHONY: all data ingest label features train test lint clean

all: $(METRICS)

data: $(RAW)
ingest: $(CLEAN)
label: $(LABELLED)
features: $(FEATURES)
train: $(METRICS)

$(RAW): $(CONFIG) $(PKG)/data/*.py
	$(PY) fraudshield.data.generate --config $(CONFIG) --out $@

$(CLEAN): $(RAW) $(PKG)/ingestion/*.py $(PKG)/pipeline/ingest.py
	$(PY) fraudshield.pipeline.ingest --config $(CONFIG)

$(LABELLED): $(CLEAN) $(PKG)/labelling/*.py $(PKG)/pipeline/label.py
	$(PY) fraudshield.pipeline.label --config $(CONFIG)

$(FEATURES): $(LABELLED) $(PKG)/features/*.py $(PKG)/pipeline/features.py
	$(PY) fraudshield.pipeline.features --config $(CONFIG)

$(METRICS): $(FEATURES) $(PKG)/training/*.py $(PKG)/evaluation/*.py $(PKG)/pipeline/train.py
	$(PY) fraudshield.pipeline.train --config $(CONFIG)

test:
	uv run pytest

lint:
	uv run ruff check src tests

clean:
	rm -rf $(RAW) data/interim data/processed models reports