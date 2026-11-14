# EXP-106 — Voice-combination completion without direct AP exposure

**Stage:** unfrozen, exploratory development pilot v0. **Not a confirmatory experiment.** The existing proposed confirmatory 107xx seeds, protected tests, and any within-event L+I evaluation remain closed.

## Scientific question

Can a small Transformer apply the familiar building blocks of active/passive voice and first/second event reference in the **unseen AP arrangement**, without ever receiving an AP source during training?

The four two-event voice arrangements are:

- **AA** — active first event, active second event.
- **PA** — passive first event, active second event.
- **AP** — active first event, passive second event (**withheld in indirect and PA-only arms**).
- **PP** — passive first event, passive second event (**introduced as indirect component evidence**).

Examples are synthetic two-event sentences with four distinct participants. A source has four queries: first/second event × agent/patient. `C_query` is the fraction of sources for which **all four** answers are correct under a four-name candidate restriction. Keep full-vocabulary accuracy and error rates alongside it.

## Intervention and matched arms

All three models receive exactly **7 AA + 7 PA common sources and 2 intervention sources per update**. Each source contributes four questions; 16 sources × 4 = **64 labels/update**. Each arm sees exactly **9 passive event occurrences/update**, including the common 7 PA events.

| Arm | Two intervention sources | Does training include AP? | Does training include PP? | Passive event positions in intervention dose |
|---|---|---|---|---|
| `pa_only` | PA + PA | No | No | first + first |
| `indirect_pp_aa` | PP + AA | **No** | Yes | first + second |
| `direct_pa_ap` | PA + AP | **Yes** | No | first + second |

The **primary precisely matched contrast** is indirect versus direct. They use the **same two source cores** (independent four-person assignments), same four gold answers per core, same modifiers, same neutral frame, identical initialization within seed, same optimizer, same common minibatches, same question count, and exactly **the same multiset of source tokens at every update**. Crucially, they have the same event-position A/P marginal frequencies: one first-event passive and one second-event passive in the dose. What changes is the *co-occurrence of the two event voices within each source*.

`pa_only` is a useful **descriptive** control for performance without second-event passive exposure. It matches total passives, total labels and shared sources but **not passive event position**, so `indirect − pa_only` alone cannot establish abstract structural transfer.

**Important unmatchable quantities:** PP+AA clusters the two passive events in one sentence whereas PA+AP distributes them; their individual input lengths and length histograms differ (3,554 of 6,400 updates in seed 10811; 3,558 in 10817). Matching source-token bags does **not** equate token positions, syntax, length, context, or learning difficulty. Those are alternative explanations, not settled controls.

## Fixed development-only protocol

- **Seeds:** 10811, 10817, both exploratory, distinct from the 107xx proposed confirmation cohort and the earlier 10801/10803 pilot seeds.
- **Training:** exact existing EXP-105 3-layer width-96 causal Transformer, four heads, AdamW LR 0.002 and weight decay 0.01, full-vocabulary cross-entropy. Each arm starts from identical weights per seed. No checkpoint-selection tuning.
- **Schedule:** 6,400 updates, two independently drawn 3,200-update sections; shared base indices imported from the original EXP-105 v3 schedule generator. Each 45-update block balances all nine familiar modifier combinations × five neutral frames for the intervention pair. The intervention pair uses *different*, matched-modifier, matched-frame cores of opposite layout parity; assignment of PP/AA versus PA/AP to cores is randomly oriented by a fixed Python RNG.
- **Development scoring:** the original **216 familiar-modifier AA/PA/AP development sources unchanged**, plus 72 **new familiar-modifier PP** dev sources sampled from existing *development* name sets (not reserved identifiers). All 288 are drawn from name sets disjoint from the training name sets. **No L+I sources are generated or scored by this EXP-106 runner.**
- **Checkpoints:** 800, 1600, 2400, 3200, 4000, 4800, 5600, **6400 fixed final**. Store final checkpoint **with optimizer states** for audit; do not retrospectively choose a best step.
- **Metrics:** per-construction `C_query` on AA/PA/AP/PP; total `C_query`; four-question accuracy; wrong-event, wrong-role and same-answer rates. Metrics are descriptive at this stage, especially given demonstrated MPS numerical variability.
- **Source protection:** `v0.partitions()` enumerates identifier-only reserved sets as it has before, but EXP-106 constructs **only train and familiar-development source texts**. It never renders reserved examples or within-event L+I examples, and never runs a protected evaluator.

## What outcomes would mean

**Promising compositional signal:** `indirect_pp_aa` consistently solves a meaningful fraction of **AP development examples despite zero AP training**, ideally beating `pa_only` and approaching the `direct_pa_ap` reference. This would demonstrate *behavioral voice-combination recombination* in the narrow grammar. It would **not yet prove an abstract role-binding mechanism** or transfer beyond this grammar family.

**No recombination:** `indirect_pp_aa` improves on PP but remains near 0% on AP. Then the missing AP combination still blocks transfer even when second-position passive syntax and its roles are trained directly in the PP context. This would be a sharp, interpretable negative result.

**Asymmetric recombination:** indirect performs well on unseen AP while direct performs poorly on unseen PP, or vice versa. The four construction cells could reveal a meaningful asymmetry in structural recombination that warrants counterfactual diagnostics.

**Optimization instability:** changes across independent MPS runs are comparable to or larger than treatment contrasts. We would need repeated runs per seed, a CPU reference or deterministic-supported backend, and paired seed-level analysis before claiming reproducibility.

## Explicit non-claims and next scientific step

Two exploratory seeds cannot support general causal or model-population inference. AP scores here are **development diagnostics**, not protected evidence. An indirect AP improvement can still reflect distributional learning of surface cues or modifier/position interactions, not genuinely abstract grammatical knowledge.

If development demonstrates basic feasibility, freeze a new independently replicated study with more seeds, matched length/position or padding controls where possible, independent lexical and structural held-outs, and a prespecified competence gate. Do not unlock the existing protected L+I evaluation simply because a development score improves.

## Commands

From the repository root, with the original EXP-105 v0–v3 source files and preflight schedule already installed:

```bash
unzip -n ~/Downloads/exp106_voice_recombination_v0_bundle.zip -d .
python3 scripts/exp106_voice_recombination_pilot_v0.py --phase prepare
python3 scripts/exp106_voice_recombination_pilot_v0.py --phase smoke --device mps
set -o pipefail
python3 -u scripts/exp106_voice_recombination_pilot_v0.py --phase run --seed 10811 --device mps | tee reports/exp106_seed10811.log
python3 scripts/exp106_voice_recombination_pilot_v0.py --phase summary
```

After inspecting the first result, repeat for seed `10817` without modifying the code or protocol. The runner refuses to overwrite existing result files. `prepare` writes only new EXP-106 audit files and validates script hashes on subsequent calls; `smoke` trains two disposable steps without writing checkpoints.

## Audit status (isolated CPU fixture)

- `prepare`: **PASS** for both 6,400-update schedules. Exact indirect/direct per-update source-token-bag matches: **6,400/6,400 per seed**.
- `smoke --device cpu`: **PASS** for three-arm identical initialization, two gradient updates, and four familiar-construction development scores.
- **No full model training was performed in the fixture.** No empirical transfer outcome is claimed yet.
