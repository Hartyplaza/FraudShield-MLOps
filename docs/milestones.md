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


## M3 — Experiment tracking and data versioning ✅

**Goal:** every model can be traced back to the exact code, data and settings that
produced it, compared with other runs, and rebuilt on demand.

### What was built

| Component | Details |
|---|---|
| Stream error analysis (`evaluation/stream_errors.py`) | Replays the live stream in-process and breaks misses down by labelling rule, with the features the model saw at the time. |
| MLflow tracking (`pipeline/tracking.py`) | Every training run logs: git commit and a `git_dirty` flag, SHA-256 of the raw data and feature matrix, generator settings, best hyperparameters, CV PR-AUC of all four candidate models, test metrics, and the full model bundle. Local SQLite store; `MLFLOW_TRACKING_URI` overrides it. Runs are named with `RUN_NAME=`. |
| DVC pipeline (`dvc.yaml`, `dvc.lock`) | The five stages with exact dependencies. The generate stage depends only on the `generator:` config section, and train only on the evaluation files it uses. `dvc.lock` pins the hash of every input and output and is committed. |
| Data store | DVC cache pushed to a local folder outside the repo (`~/dvc-store`, configured in the untracked `.dvc/config.local`). Data never enters Git or GitHub. |
| `reports/metrics.json` in Git | Each commit carries its metrics, so `dvc metrics diff <branch>` compares versions. |
| `is_risky_type` feature | Added after the error analysis (below). Batch and online versions are covered by the streaming = batch test; 72 tests. |

### Finding: all 9 missed frauds came from one rule

The M2 stream missed 9 of 75 frauds. Breaking the misses down by labelling rule:

| Rule | Frauds | Caught | Missed |
|---|---|---|---|
| A only (night + high amount + cashout/transfer/withdrawal) | 37 | 28 | 9 |
| C only (high amount + device switch + no location) | 29 | 29 | 0 |
| Combinations | 9 | 9 | 0 |

Rule A is the only rule that depends on the transaction type. The model saw the type
only as an integer code (`cashout=0 … transfer=6, withdrawal=7`), so the three risky
types are not contiguous, and fewer than 400 training rows are both night and high amount.
The model settled for "night + high amount", which explains both the misses and the
false alarms: 18 of 19 false alarms had two of a rule's three signals, mostly the right
time and amount with the wrong type. Lowering the threshold would not help, because it
would also flag the wrong types.

Fix: an explicit `is_risky_type` flag, logged as a second MLflow run and compared with
the baseline.

| | baseline | risky-type |
|---|---|---|
| Features | 28 | 29 |
| Test PR-AUC | 0.9384 | 0.9910 |
| Test TP / FP / FN / TN | 56 / 21 / 5 / 1466 | 60 / 6 / 1 / 1481 |
| Threshold | 0.77 | 0.91 |
| Stream: caught / missed / false alarms | 66 / 9 / 19 | 72 / 3 / 5 |

Caveat: the labels come from rules, so a feature that mirrors a rule's input is bound
to help. The general lesson holds on real data: when risky categories are not contiguous
in an integer encoding, give the model a domain flag, one-hot encoding or native
categorical handling.

### Is the improvement real? Seed variation

A second dataset (generator seed 7, same size) was built and compared with
`dvc metrics diff`:

| | seed 42 | seed 7 |
|---|---|---|
| PR-AUC | 0.991 | 0.992 |
| Recall | 0.984 | 0.926 |
| Precision | 0.909 | 0.943 |

PR-AUC (ranking quality) is stable to about 0.001. Threshold metrics move several points
because the test sets hold only 54–61 frauds, where one fraud is worth ~2 points of
recall, and the threshold itself is tuned on a validation set of ~50. The `is_risky_type`
gain (+0.053 PR-AUC) is roughly 50 times the seed-to-seed variation, so it is real; a
gain of 0.01 would not have been distinguishable from noise.

> **Correction (M4):** two seeds were not enough. Across 8 datasets PR-AUC varies by
> about ±0.02, not ±0.001. A paired comparison on the same 8 datasets confirms the
> `is_risky_type` gain (+0.048, better on 8 of 8). See M4, finding 4.

### Evidence of reproducibility

- The baseline run reproduced the M2 model exactly (PR-AUC 0.9384, 56/21/5/1466), and
  its data hashes (`6f7d4dfd…` raw, `1d9a7ddc…` features) matched across two machines.
- `dvc repro` rebuilt the risky-type model identically (0.991, 60/6/1/1481); a second
  `repro` skipped every stage.
- After deleting all data and the model, `dvc pull` restored them byte for byte
  (`6f7d4dfd`, `8b696e59`), matching the hashes recorded in MLflow.
- Switching from the seed-7 branch back to `m3-mlflow`, `dvc checkout` restored the
  seed-42 data and model from cache in under a second, with no retraining.

### Problems hit

**1. An edit script that was never run.** Training ran, but no MLflow run appeared. The
script that wires in `log_run` had not been executed; training ran only because Make
also watched the whole `evaluation/` folder, where a new file had appeared. Two lessons:
check that a step's expected output appeared, and dependencies that are too broad
trigger rebuilds for the wrong reasons. The DVC train stage lists only the files it uses.

**2. Optional extras are not installed by default.** `ModuleNotFoundError: mlflow` after
adding it to the `train` extra. `uv sync --all-extras` installs it.

**3. Work done on the wrong branch.** The first MLflow run showed `main` as its source
branch, meaning the feature branch had never been created. `git switch -c` carried the
uncommitted changes onto a new branch before committing.

**4. A run before the commit is marked dirty.** The baseline run is tagged
`git_dirty: true` because it used uncommitted code. Later experiments were committed
first, then trained, so their runs point at an exact commit.

**5. MLflow UI port already in use.** An earlier UI was still running; it reads the same
database, so refreshing the page was enough.

### Known issues (carried forward)

- **Two pipeline definitions.** The Makefile (used by the Docker training image) and
  `dvc.yaml` describe the same DAG and can drift apart.
- **Docker training runs are not tracked.** The container logs to its own `mlflow.db`,
  which is discarded on exit. Needs a shared tracking server (M6).
- **Tracking and data store are local.** Fine for one person; a team needs a remote
  tracking server and a cloud DVC remote.
- **The model is logged as plain artifacts**, not as an MLflow model, so it cannot be
  registered or promoted yet (M5).
- **Run duration is misleading.** The MLflow run opens after training, so it times the
  logging (~130 ms), not the 3–4 minute search.
- **Single-seed evaluation.** Comparisons rest on one split of one dataset. Decisions
  should use several seeds or confidence intervals (M4).

## M4 — Data validation and ML-specific tests ✅

**Goal:** bad data stops the pipeline before it reaches a model, and a model that is
worse, broken or leaning on the wrong signals is rejected before it is accepted.

### What was built

| Component | Details |
|---|---|
| Data contracts (`validation/schemas.py`, pandera) | Checked after ingest, label and features. Row checks: types, `user_id` pattern, known transaction types, currencies and log formats, amount in [0.01, 10000], binary flags, non-negative counts, NaN only where expected. Batch checks: at least 1,000 rows, at least 60% of raw lines parsed, null rates, fraud rate in [1%, 10%]. Limits live in `configs/pipeline.yaml -> validation`. |
| Quality gate stage (`pipeline/gate.py`) | Runs after training. Metric limits (PR-AUC ≥ 0.95, recall ≥ 0.90, precision ≥ 0.85, FPR ≤ 1%), 8 golden cases scored end to end from raw log lines, 4 invariance checks and 4 directional checks. Its own DVC stage, so changing a limit re-checks the model in seconds without retraining. |
| Threshold policy | Highest recall with out-of-fold precision ≥ 0.90 (`training.min_precision`), tuned on cross-validated predictions over all 243 training frauds instead of F1 on 49 validation frauds. |
| Seed robustness stage (`pipeline/seeds.py`) | Refits the model recipe on 8 freshly generated datasets and reports mean, spread and worst case per metric (`reports/seeds.json`). |
| Tests | 98 total, including 16 that each feed one kind of bad batch and assert the *specific* check that stops it. |

Pipeline: `generate → ingest → label → features → train → gate → seeds`, with
validation inside the first three data stages.

### Finding 1: bad data is stopped where it enters

Drill: raising the generator's malformed-line rate from 11% to 50%. Ingest stopped with
`only 39.5% of raw rows parsed (minimum 60%)`; label, features and training never ran
and the model on disk was untouched. `git checkout` of the config and `dvc.lock` plus
`dvc checkout` restored the previous state.

### Finding 2: the gate would have rejected the M2 model

Pointing the gate at the M2 baseline (from its MLflow artifacts): fails PR-AUC (0.938),
precision (0.727) and FPR (1.4%). It still passed every golden case; the £4,800 night
purchase scored 0.495, much closer to the line than now. Metric limits catch a worse
model; golden and behaviour cases catch a broken or misguided one.

### Finding 3: the model was reading the log format

Invariance check: the same transaction written in each of 7 log formats and 3
currencies. Decisions never changed, but for a £4,800 night purchase the score ranged
from 0.025 to 0.334 depending only on format and currency, enough to flip the API's
`risk_level` between low and medium. The fraud rules never use either field, but both
were model inputs (`log_format_encoded`, `currency_encoded`).

Fix: drop both features (29 → 27). Every invariance spread is now exactly 0.0000.

Removing them exposed a second problem. The retrained model was better (PR-AUC 0.993)
but failed the precision gate (0.82): the F1-tuned threshold picked 0.80 over 0.85 on
the strength of one transaction among 49 validation frauds, costing 7 extra false
alarms. Replacing it with a stated precision target on out-of-fold predictions gave a
threshold of 0.90 and passed all 20 checks.

| | M3 model | M4 model |
|---|---|---|
| Features | 29 | 27 |
| Threshold rule | best F1 on 49 validation frauds | highest recall with precision ≥ 0.90, out-of-fold over 243 frauds |
| Format/currency score spread | 0.31 | 0.00 |
| Test PR-AUC (seed 42) | 0.991 | 0.993 |
| Test TP / FP / FN | 60 / 6 / 1 | 59 / 6 / 2 |
| Live stream: caught / false alarms | 72 / 5 | 72 / 6 |
| Quality gate | fails 1 of 20 | passes 20 of 20 |

### Finding 4: the M3 noise estimate was wrong

Across 8 new datasets, the current recipe scores:

| | mean | ± sd | worst |
|---|---|---|---|
| PR-AUC | 0.965 | 0.019 | 0.928 |
| Recall | 0.905 | 0.062 | 0.815 |
| Precision | 0.900 | 0.044 | 0.833 |

Seed 42 (0.993) sits at the lucky end, partly because the hyperparameters were tuned on
it. M3 called PR-AUC "stable to about 0.001" from two seeds that happened to agree; the
real spread is about ±0.02. Re-testing both feature changes as paired comparisons on
the same 8 datasets:

| Change | Mean PR-AUC change | Better on | Verdict |
|---|---|---|---|
| `is_risky_type` added (M3) | +0.048 | 8 of 8 seeds | Real: ~7× the noise |
| Format and currency removed (M4) | −0.002 | 3 of 8 seeds | No measurable accuracy effect; justified by invariance |

Lesson: compare changes on the same set of datasets (paired), never one run against
another.

### Problems hit

**1. A commit without its wiring.** The gate's files were committed, but the script that
adds its config and DVC stage was never run; tests failed with `KeyError: 'gates'` and
the next commands ran anyway because each was on its own line. Fixed with a follow-up
commit; later steps chained commands with `&&` so a failure stops the chain.

**2. Missing columns were hard to read.** pandera reports a missing column at batch
level, with the name only in the failure value. The report now names the column.

**3. A failed gate writes `gate.json`.** Committing it would put a failed result in Git;
it is reset with `git checkout -- reports/gate.json` before committing, so Git only
holds results from passing runs.

### Known issues (carried forward)

- **Gate limits are judged on one dataset.** The recall limit (≥ 0.90) passes on seed 42
  but this recipe would fail it on 3 of 8 seeds. Promotion (M5) should compare a
  challenger with the current champion across seeds, not one split.
- **Docker skips the gate and seeds stages.** The training image runs `make all`, which
  includes validation (it lives inside the stage scripts) but not the two new stages;
  the Makefile and `dvc.yaml` have drifted further apart.
- **Golden and behaviour cases are hand-written** and cover the known rules; they say
  nothing about fraud patterns nobody wrote down.
- **The seed stage reuses seed-42 hyperparameters**, so it measures the variance of this
  exact recipe, not of the whole training procedure (which would include the search).
- **`/predict` still accepts `log_format_encoded` and `currency_encoded`**; the model
  ignores them, but the request schema should drop them.