# Milestones

## M1 — Reproducible build ✅

**Goal:** anyone can clone the repo and get the same trained model and a running
fraud API with one command, on any machine, without access to the private dataset.

### What was built

| Component | Details |
|---|---|
| Packaging | `pyproject.toml` (src layout) with separate dependency groups: `core`, `serve`, `train`, `dev`. Exact versions pinned in `uv.lock`. |
| Synthetic data generator | Statistical twin of the original assessment file (which cannot be published). Seeded and config-driven; renders all 7 raw log formats. |
| Pipeline | Four stages (ingest → label → features → train), each a small CLI reading paths from `configs/pipeline.yaml`. |
| Makefile | Encodes the stage dependency graph; `make all` reruns only stages whose inputs changed. |
| Docker | One multi-stage Dockerfile producing a `train` image and a `serve` image. Frozen dependency install, non-root user, healthcheck. |
| Compose | `train` runs once and writes the model to a shared volume; `api` starts only after training succeeds and mounts the model read-only. |
| Tests | 35 tests: 20 parser/renderer contract tests, 8 generator tests, 7 API tests. |

**Run it:** `docker compose up` → trained model + API on `localhost:8000`.

### Results (generated data, seed 42)

| Metric | Value |
|---|---|
| Best model | LightGBM |
| PR-AUC | 0.9468 |
| ROC-AUC | 0.9971 |
| Recall / Precision | 0.869 / 0.883 |
| False positive rate | 0.0047 |
| TP / FP / FN / TN | 53 / 7 / 8 / 1480 |

The generator matches the original data's profile: parse rate 77.4% (original 77.7%),
fraud rate 3.9% (original 3.4%), all 7 log formats present.

### Evidence of reproducibility

The same model was produced, bit for bit, in three different environments:

| Environment | Python setup | PR-AUC | TP/FP/FN/TN |
|---|---|---|---|
| Independent environment | pip-installed packages | 0.9468 | 53/7/8/1480 |
| Laptop (WSL2 Ubuntu) | `uv` venv from lockfile | 0.9468 | 53/7/8/1480 |
| Docker container | Debian slim, frozen `uv` install | 0.9468 | 53/7/8/1480 |

The generator also produced byte-identical data on two machines from the same seed,
and `/predict` returned identical scores (0.9777) inside and outside the container.

### Problems hit and how they were solved

**1. Plotting library leaking into the serving dependencies.**
`shap_analysis.py` imported matplotlib at module level, so the API could not start
without a plotting library. Found by installing *only* the serve dependencies
(installing all extras had hidden it). Fixed by moving the import into the one
function that plots.

**2. Generated log lines that parsed but carried wrong values.**
The first renderer hardcoded currency symbols copied from example lines. Every line
still parsed, so a quick check passed, but the currency was wrong in 4 of 7 formats.
Caught by round-trip contract tests that compare every field, not just whether
parsing succeeds.

**3. Training stalled for 20+ minutes (nested parallelism).**
`RandomizedSearchCV(n_jobs=-1)` ran models that also used `n_jobs=-1`, so dozens of
threads competed for a few cores. Fixed by parallelising only at the search level.
LightGBM search: stalled → 39 s; whole training stage: 4 min 24 s.

**4. Deprecated scikit-learn API.**
`LogisticRegression(penalty=...)` is deprecated in scikit-learn 1.8 and removed in
1.10. The lockfile meant this produced warnings rather than a crash; replaced with
`l1_ratio`. Side effect noted: the search space changed order, so the Logistic
Regression CV score moved slightly (0.6207 → 0.6244). Even "pure refactors" can
change results.

**5. API could not load the model in the serving container.**
The saved model is an imbalanced-learn `Pipeline` (it includes the ADASYN step used
in training). Unpickling it requires `imblearn` to be installed, even though serving
never resamples. Found by loading the model with serve-only dependencies; fixed by
adding `imbalanced-learn` to the serve group.

**6. Single-item batch requests crashed the API (HTTP 500).**
`FraudPredictor.predict` returned scalars for one row and lists otherwise, so
`/predict/batch` with one transaction raised `TypeError`. Found by an API test;
fixed by making `predict` always return lists.

**7. Docker builds extremely slow on WSL2.**
The first image build took ~100 minutes; downloads ran at 14–120 KB/s through
Docker's bridge network. Building with `--network host` brought the train image
build down to 5.5 minutes, pointing to the bridge network as the bottleneck.

### Known issues (carried forward)

- **Serve image is 973 MB** (228 MB compressed). SHAP pulls in `numba` and `llvmlite`
  (~190 MB). Planned fix: use LightGBM's built-in `pred_contrib` for explanations
  and drop `shap` from serving.
- **The saved model includes training-only steps** (ADASYN), coupling serving to
  training libraries. Planned fix: save only the fitted classifier.
- **The model leans on `is_weekend`**, which is not part of any fraud rule
  (SHAP −2.36 on a test transaction). To investigate with model behaviour tests in M3.
- **Deprecation warnings**: FastAPI `on_event` (use `lifespan`), Pydantic
  class-based `Config` (use `ConfigDict`), Starlette `httpx` test client.
- **Every pipeline stage depends on the whole config file**, so any config change
  reruns everything. To be addressed with DVC parameter tracking in M2.