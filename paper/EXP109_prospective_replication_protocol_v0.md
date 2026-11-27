# EXP-109 — Prospective within-grammar replication of the AP construction gap

**Status:** implementation + independent-evaluation commitment prepared, **not yet trained or scored**. This document must be committed together with the exact Python source and `reports/exp109/replication_v0/manifest.json` **before the first EXP-109 `--phase run`**. It is *not* an external registry entry or independent-grammar validation.

## Prior evidence and motivation

Exploratory EXP-106–108 showed large, variable differences between an original AP sentence and a literal PA clause-reordering of the **same two clauses**. The baseline PA-only model had AP 25% / PA literal 83.3% in exploratory seed 10811 and AP 0% / PA literal 88.9% in 10817. The indirect PP+AA model acquired AP accuracy without seeing AP in training (50% and 93.1% respectively), but numerical optimization variability is substantial. All 72 historical AP development anchors have been repeatedly examined and are **not** eligible as new independent test examples.

## Questions, registered hierarchy, and experimental unit

**Primary:** for the `pa_only` training arm, what is the paired **literal clause-order gap**, `C_query(PA_literal) − C_query(AP)`? Each example uses identical clauses/tokens, with first/second references and corresponding gold answers switched. The experiment tests whether the baseline systematically fails on AP while solving an identical-content PA arrangement.

**Key secondary (NOT an isolated causal mechanism):** `C_query(AP | indirect_pp_aa) − C_query(AP | pa_only)`. Both arms receive identical common examples, source identities, four gold answers per source, and 9 passive events per update, but the indirect arm has second-event passive evidence from PP while the baseline does not. A positive contrast alone cannot prove an abstract compositional mechanism.

**Descriptive controls:** the `direct_pa_ap` arm shares the indirect arm's per-update aggregate token multiset, gold labels, core identities and event-position voice *marginals*, but differs in passive co-occurrence, exact positions and sentence-length distributions. The direct arm is an anchor for attainable competence, not an inference of superiority. Evaluate original AP, literal PA, AA, PP, AP participant renaming and AP second-event role reversal for all three arms; report full-vocabulary accuracy, wrong-event/wrong-role/copying rates, second-event pair classes and per-run AA/PA competence.

**Experimental units:** independent *training seeds*, not sentences and not numerical repetitions. Prespecified `10901, 10903, 10907, 10909, 10913, 10919, 10921, 10927`. Two fresh-process repetitions per seed. Eight seed-level differences are the units for inference. MPS non-bitwise determinism is known and must be reported.

## Training and inference freeze

- Exactly three arms from EXP-106: `pa_only = 7 AA + 7 PA + 2 PA`, `indirect_pp_aa = 7 AA + 7 PA + PP + AA`, `direct_pa_ap = 7 AA + 7 PA + PA + AP` each update. Each source contributes 4 labels; 64 labels/update.
- Same EXP-105 small causal Transformer: 3 layers, width 96, 4 heads, AdamW LR 0.002, weight decay 0.01, no dropout; full-vocabulary cross-entropy.
- `6,400` updates per seed/repeat, **no best-checkpoint selection, early stopping, discarded runs or selective reruns**. Runs record losses only at steps 800–6400, and score the new evaluation exactly once at the fixed endpoint. Checkpoints are retained whether the arm learns or fails.
- Two numerical repetitions of each identical seed/schedule start in **separate processes** on MPS. If unavailable/hardware changes, document as a protocol deviation; do not silently switch the registered device.
- New train seeds use the EXP-106 schedule algorithm and identical common and intervention source indices across arms; all seed schedules and token-bag audits are generated before test evaluation. The previously proposed `107xx` cohort is untouched.

## Independent evaluation construction and integrity

- Existing **non-reserved** 100 development four-name sets from the EXP-105 partition are used. The original 95 reserved identifier sets, protected tests and within-event L+I compositions remain closed. No protected sentences are generated, read, logged or scored.
- For each development name set, generate new, deterministic participant **orderings excluded from the two historical development core orderings**. Construct a new AP anchor for each of the 9 familiar (bare/L/I) modifier pairs; sample **16 anchors per modifier pair**, for 144 new AP anchors. This is a new **sentence-core/surface split**, not novel lexical items, a new grammar, or entirely new name sets.
- Each anchor produces 6 predefined presentations: (1) original AP, (2) **literal PA clause swap** preserving every source token and clause's internal structure, (3) AA by flipping the second event voice, (4) PP by flipping the first event voice, (5) cyclic participant rename, (6) second-event agent/patient reversal. This gives 864 scored presentations per checkpoint.
- An independent surface parser validates every presentation and semantic answer. The generator checks that no surfaces overlap the 288 original EXP-106 development sources; 864/864 surfaces were distinct in the preparation audit. The evaluator never displays examples before the freeze, and stores only corpus/hash metadata and post-score aggregate results.
- **Important scope:** earlier experiment selection already used the same synthetic grammar and dev partition. Consequently, this is a *prospectively held-out within-family replication*, not an independent external/structural generalization benchmark.

## Locked analysis and decision rules

1. For each of the 16 runs, compute the paired AP/PA-literal four-name-candidate `C_query` gap in `pa_only`. Compute the secondary indirect-minus-baseline AP `C_query` difference using the same anchor set.
2. Average the two numerical-repetition contrasts **within seed**, then calculate the unweighted mean and sample SD over the 8 seeds. Report a two-sided 95% t interval with `df=7`, critical value `2.364624251`. This interval is descriptive given small n and training instability, not evidence of universal architecture-wide behavior. Do not treat 144 sentences or 16 repetitions as independent training replicates.
3. Report all 16 outcomes and aggregate diagnostics, including negative, zero and non-converged runs. Predefined **competence descriptor**, not an exclusion criterion: AA and PA-literal `C_query ≥ 0.75` within a run/arm. Do not select successful checkpoints or condition the primary effect on passing the descriptor.
4. No stopping or adaptation based on intermediate test results. The `summary` routine refuses to report an overall effect until all 16 prespecified run files are present.
5. Keep current EXP-107/108 results as exploratory context; do not combine them with EXP-109 as if they were unseen confirmatory data. Do not access reserved L+I tests on the strength of this result.

## Controls and residual confounding

The literal AP-to-PA comparison holds token multiset, total length, lexical identity, names, event meanings and clause-internal syntax fixed, while altering clause and question-reference positions **by design**. It cannot distinguish grammatical order sensitivity from position-based heuristics by itself.

The `pa_only` versus `indirect_pp_aa` intervention differs in **second-event passive exposure and source-level co-occurrence**, not solely in abstract grammatical composition. The `indirect` versus `direct` match controls total token bags and event-position voice margins but not per-source length distributions. EXP-109 is therefore a stringent replication of a *behavioral pattern*, not an isolating causal mechanism test.

## Freeze and use

1. Install the bundle into the **`~/Developer/exp102-clean-design` worktree** using its existing active Python environment.
2. Run `--phase prepare` and `--phase smoke --device mps`. Neither scores the independent evaluation.
3. Review `paper/EXP109_prospective_replication_protocol_v0.md`, `scripts/exp109_replication_v0.py`, `reports/exp109/replication_v0/manifest.json`. Commit those three exact files locally **before any `run`**. `run` verifies the script and manifest match a committed Git ancestor; protocol hashes also pin dependency source files and evaluation generation.
4. Run each seed/repeat once in a fresh process, save logs, preserve saved results. Run `--phase summary` only after all 16 files exist. Any implementation changes **after** viewing evaluation outcomes must be reported as protocol amendments and require genuinely new held-out material.

**What is not claimed:** completed EXP-109 data, independent syntactic-family transfer, reliable training effects, or a neural mechanism. Those require results and future studies.
