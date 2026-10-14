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
  - **The API cannot score a raw log line.** Category encodings are recomputed from
  whatever data is present (not saved with the model), and velocity/user features
  need per-user history. Scoring one line alone yields different features than
  training (training/serving skew). Planned: persist encoders (M3), feature store
  and raw-log endpoint (M5).
  - **The API cannot score a raw log line**: category codes are recomputed per batch and
  history features need per-user state (training/serving skew). Being fixed in M2.
- **Batch scoring of a CSV depends on what else is in the file**, because features are
  computed within the file. Being fixed in M2.



## M2 — Real-time scoring ✅

**Goal:** a raw transaction log line arrives, the system scores it immediately using
that user's history, and decides fraud or legitimate — with no difference between how
features are computed in training and in production.

### What was built

| Component | Details |
|---|---|
| `FeatureState` (`features/online.py`) | Computes the 28 features for one transaction from the user's history so far, then adds it to that history. Welford running mean/std, 30-day deque for velocity windows, last device/location/time. |
| `CategoryEncoder` | Fixed category → code mapping learned once and saved with the model. Unseen values map to -1 instead of shifting other codes. |
| Leak-free training | The features stage replays transactions in time order through `FeatureState` — the same code the API runs. |
| Model bundle | `models/` holds model, threshold, feature list, encoders and user state; deployed as one unit. |
| `POST /score` | Accepts a raw log line in any of the 7 formats; returns the decision and updates the user's history. Thread-safe state updates. |
| Stream demo | Replays unseen transactions into the running API one at a time and reports a live tally. |
| Tests | 71 total, including *streaming one-by-one = batch replay* (identical probabilities). |

### Finding: the original features leaked the future

Per-user statistics (`user_mean_amount`, `user_std_amount`, `user_max_amount`,
`amount_z_score`, `amount_vs_user_max`) were computed over each user's *entire* history,
including transactions that had not happened yet. Proof: batch and online values agree
**only on each user's last transaction** — the one point where "all history" and
"history so far" coincide. A test now pins this down.

The other 23 features matched exactly (100% of rows).

| | Leaky features | Leak-free features |
|---|---|---|
| CV PR-AUC | 0.9065 | 0.9145 |
| Test PR-AUC | 0.9468 | 0.9384 |
| Recall / Precision | 0.869 / 0.883 | 0.918 / 0.727 |

The leak barely mattered here because the fraud rules depend on single-transaction
signals, not user history; differences are within noise (61 test frauds). On real data,
where history matters, a leak like this can inflate offline scores and then collapse in
production. It is now impossible by construction.

### Live results

Two weeks of new, unseen transactions (starting after the training period, different
seed), streamed one at a time:

| Metric | Value |
|---|---|
| Frauds caught | 66 of 75 (recall 0.88) |
| False alarms | 19 (precision 0.78) |
| Latency | p50 ~7 ms, p95 ~10–14 ms |

The Docker deployment — training its own model from scratch — produced **identical
decisions** (66 / 9 / 19) to the laptop setup.

### Problems hit

**1. API could not score a raw line.** Category codes were recomputed per batch (a
one-row batch encoded everything as 0) and history features had no history. Solved by
the saved encoder and the stateful `FeatureState`.

**2. A stale process answered instead of the container.** The demo ran while a local
API from an earlier test still held port 8000; the container API was created but never
started, and the demo silently hit the old process. Found with `docker compose ps -a`.
The container's port binding also failed to attach and needed `--force-recreate`.

**3. That accident exposed a real design gap: scoring is not idempotent.** The old
process had already seen the same two weeks, so every transaction was added to user
history twice — inflating velocity features and raising false alarms from 19 to 25.

### Known limitations

- **State is in memory.** Restarting the API resets history to the end of training;
  multiple replicas would each hold a different copy. Fix: shared store (e.g. Redis).
- **Transactions are assumed to arrive in time order.** A late, older transaction would
  be treated as the newest.
- **Not idempotent.** Re-sending a transaction adds it to history again. Real streams
  deliver at-least-once; production needs a transaction ID and de-duplication.
- **Threshold (0.77) is chosen by F1 on a small validation set** — in practice it is a
  business decision balancing fraud losses against review capacity.