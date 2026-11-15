# EXP-105: PA/AP development learnability pilot v0

**Status:** disposable, exploratory, **not** the confirmatory EXP-105 protocol. No reserved examples, protected evaluation, or L+I scoring. This stage tests implementation and base learnability. Do not use the pilot seed in a later confirmatory panel.

## Scientific and engineering question

Do two identical-initialization small causal Transformers train successfully under the audited v3 update schedule, and can we measure whether PA-vs-AP exposure is associated with improved AP comprehension on separate **familiar-syntax development** name sets? This pilot is **not a test of L+I structural transfer**.

## Conditions and controls

| Item | Pilot v0 |
|---|---|
| Pilot seed | `10801`, deliberately not one of the eight proposed 107xx confirmatory seeds |
| Shared base per update | 7 AA + 7 PA sources |
| Familiar exposure dose | 2 PA sources |
| Target exposure dose | 2 AP sources, paired with familiar dose on semantic core and four answers |
| Supervision | Four questions per source; 64 labels/update; equal full-vocabulary CE |
| Sampling | **Direct import** of `exp105_training_schedule_preflight_v3.iter_schedule`; no reimplemented RNG |
| Network | 3-layer, width-96, 4-head causal attention, no dropout, 48 absolute positions |
| Optimizer | AdamW, LR=.002, weight_decay=.01, independent optimizer state per arm |
| Evaluation | Four-name forced choice, full-vocabulary accuracy, C_query, AP/PA/AA C_query, wrong-event, wrong-role, and same-answer rates |
| Dev sample | 8 sources per voice × modifier cell = 216 familiar-syntax sources, fixed selection seed 105450; **no LI** |
| Pilot steps | 45, 90, 180, 270, 360, 450 (longer options available but require review before use) |
| Early stopping | None; report the chosen fixed pilot budget, do not pick best milestone |
| Outputs | New report and paired checkpoint under `reports/exp105/development_pilot_v0/` and `results/exp105/development_pilot_v0/`; refuse overwrite |

## Required integrity checks

1. Read the **already written** v3 schedule report (`reports/exp105/preflight/training_schedule_preflight_v3.json`); check that the source-file hashes, split digest and schedule configuration still agree.
2. Build the vocabulary from **training only**, including the neutral temporal frames; verify development familiar-syntax token coverage.
3. Confirm familiar and target model parameters start identical, and dose data have the same gold four-question answers and candidate names.
4. Use `prepare` before `smoke` or `run`; the pilot protocol binds code hashes, data selection and vocabulary, and refuses altered dependencies.
5. Never load, score, unlock or create reserved examples. The inherited preflight source pool construction uses only **development** L+I sources to check text disjointness, not to evaluate models.
6. No optimizer checkpoint selection based on development outcomes. No claims of independent replication, mediation or a unique mechanism.

## Commands (from the repository root)

```bash
cp ~/Downloads/exp105_development_pilot_v0.py scripts/
python3 scripts/exp105_development_pilot_v0.py --phase prepare
python3 scripts/exp105_development_pilot_v0.py --phase smoke --device mps
```

Only after the two-phase smoke check passes:

```bash
set -o pipefail
python3 -u scripts/exp105_development_pilot_v0.py \
  --phase run --steps 450 --device mps \
  | tee reports/exp105_pilot_seed10801.log
```

The first data-bearing run uses only pilot seed 10801. The proposed prospective seed set (10701, 10703, 10709, 10713, 10721, 10727, 10729, 10733) is intentionally untouched.

## What would count as useful output

- The two training losses decline and model scoring remains finite and correctly decomposed.
- Base AA/PA learning is assessed on dev at fixed milestones, without choosing the best checkpoint.
- AP development differences are **exploratory feasibility diagnostics**, not confirmation of a structural generalization effect.
- If competence is inadequate, record the failure and revise **using only new development pilots**. Freeze a new confirmatory protocol before using any 107xx seeds or examining independent reserved data.

## Limitations carried forward

- Passive location changes surface position, word order and local syntactic context. The relative contribution of each remains open.
- Although token multisets match per update and source-length distributions balance per complete 45-step window, individual-update length histograms and token-position distributions differ.
- The experiment uses the same synthetic task family as EXP-102; it is not an independent grammar replication.
- The full L+I structural generalization protocol and competence gate remain to be designed and frozen.
