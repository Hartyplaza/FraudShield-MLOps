# Milestones

## M1 — Reproducible build (in progress)

### Built
- (what exists now: packaging, generator, pipeline, Makefile, tests)

### Problems hit and how they were solved
- (matplotlib leaking into serving dependencies)
- (hardcoded currencies that parsed but were wrong; caught by round-trip tests)
- (nested parallelism stalling LightGBM; 20+ min -> 39s)

### Evidence of reproducibility
- (same seed -> byte-identical data on two machines)
- (same metrics on two machines: PR-AUC 0.9468, TP/FP/FN/TN 53/7/8/1480)