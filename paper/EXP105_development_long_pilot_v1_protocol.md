# EXP-105: Extended development learnability pilot v1

**Status:** Exploratory and development-only. This is not the prospective confirmatory EXP-105 protocol.

## Motivation

The first exploratory seed (10801) reached only 25.46% / 31.48% overall C_query at step 3200 for PA-dose familiar / AP-dose target arms. The construction-specific audit found targeted AP event membership improves substantially, but same-answer collapse remains high and PA performance deteriorates. Neither model established stable familiar AA/PA competence. Before drawing a mechanism or transfer conclusion, test whether this reflects limited training duration and between-seed variability.

## Paired design

- Seeds: **10801 and 10803**, development-only. The original proposed **107xx** confirmatory seeds remain unused.
- Both arms are identically initialized for a given seed; both use the same 3-layer width-96, 4-head causal Transformer and AdamW (`lr=0.002`, `weight_decay=0.01`).
- Training per update: 7 AA + 7 PA common sources, 2 paired PA-versus-AP dose sources, four complementary questions/source, 64 labels; full-vocabulary cross-entropy.
- **Fixed horizon: 6400 updates.** First 3200 reuse the original v3 deterministic schedule, while the next 3200 repeat the v3 **generator algorithm** with a different predetermined RNG seed (`seed + 100000`). Each half balances full 45-update blocks. The per-update matched token-multiset and source/gold invariants are audited.
- Model parameters are **retrained from scratch**, not resumed from the saved step-3200 `.pt` file (the original did not archive optimizer state). Do not mistake this for a continuous checkpoint extension.
- Development scoring only: the same 216 AA/PA/AP familiar-syntax development sources and hashes as v0; no L+I scoring and no reserved/protected evaluation. Note that the inherited v3 pool builder renders the existing development L+I *design* pool for disjointness checks but does not use it for model scoring.
- Preselected milestones: 800, 1200, 1600, 2400, 3200, 4000, 4800, 5600, 6400. Report both arm per-construction `C_query`, overall accuracy, wrong-event, wrong-role and same-answer rates. Fixed final checkpoint; never select the best development milestone as a publication endpoint.
- Fresh records in `reports/exp105/development_long_pilot_v1/`; checkpoints in `results/exp105/development_long_pilot_v1/`. No overwrites of prior files.

## Decision guide (not a confirmatory gate)

1. If both arms learn familiar AA **and** PA reliably, begin drafting a new confirmatory protocol with a strict familiar competence gate and an independent structural transfer test.
2. If only one arm learns those constructions, diagnose exposure-induced interference / optimization imbalance on development data. Do not interpret an AP advantage as general structural competence.
3. If neither arm learns AA/PA, revisit the feasibility of the grammar, sampling and training configuration *using development-only experiments*. Keep prior results immutable.
4. Inspect AP event-recognition, reversal and collapse conditional on correct-event selection, rather than interpreting one aggregate accuracy as a mechanism.

This pilot estimates learnability and seed sensitivity; it does not independently replicate an L+I transfer effect. The same training-family, word-order, position-distribution and per-update length limitations documented in v3 continue to apply.
