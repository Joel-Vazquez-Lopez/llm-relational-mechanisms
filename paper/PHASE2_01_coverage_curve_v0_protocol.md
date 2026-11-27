# PHASE2-01 — Matched structural-exposure coverage curve

**Status:** prospective, fixed v0 experimental design for a **new Phase 2 study**, informed by the completed EXP-109–116 research sequence. Not externally preregistered. The exact code, launcher, this protocol and generated audit manifest must be committed together **before any training or evaluation**. Protected/reserved and L+I evaluations remain unopened.

## Motivation and distinction from Phase 1

The original series established a reproducible construction-sensitive failure in 3-block scratch-trained causal Transformers. EXP-114 independently replicated sensitivity to input positional codes and first-block updates on new seeds and disjoint *within-grammar* examples. EXP-115–116 showed that broad and participant-state swaps can move wrong-role decisions toward correct answers **and** toward wrong-event answers, without a unique role-specific head or selective repair. Those data are already examined and do not justify another mechanism-selection sweep.

**New question:** Holding the *amount of supervised data*, per-update gold labels, overall token frequencies and passive-event position marginals constant, how does direct exposure to the combined construction `PA + AP`, rather than separate `PP + AA` evidence, affect generalization to newly sampled AP sentences?

This tests **structural co-occurrence/coverage during training**, not a unique syntactic circuit. The difference between the two matched dose pairs necessarily includes within-source **voice combinations and source-length/position distributions**; it cannot be interpreted as varying one semantic concept in isolation.

Relevant context includes Ahuja et al., *Learning Syntax Without Planting Trees* (TACL 2025); *From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers* (NeurIPS 2025); and recent work on positional pointers in compositional generalization. Their questions are related, not evidence that PHASE2-01 has already been answered.

## Frozen training design

- Architecture, vocabulary, optimizer, regularization, learning rate, maximum tokens and training objective: **exact EXP-109/114** setup (width 96, 3 causal blocks, 4 attention heads); 6,400 updates, fixed endpoint, no checkpoint selection.
- **Eight new independent seeds:** `11701, 11703, 11707, 11709, 11713, 11719, 11721, 11727`; **two numerical MPS repeats** per seed; five dose arms starting from identical within-seed model weights. Total: **16 runs, 80 trained models**. No seed, model or repeat exclusion for slow learning or poor competence.
- Each training update contains **14 identical background sources** (seven AA, seven PA), plus **two matched dose sources**. All arms have 16 source examples and 64 supervised answers per update.
- The dose controls are `PP` on selected core A + `AA` on selected core B (indirect) versus `PA` on A + `AP` on B (direct). The source cores, gold answers, modifier/frame stratum and aggregate per-update token bag are exactly shared. Both pairs have exactly one passive first event and one passive second event. Individual sentence length/voice conjunctions may differ.
- Each arm takes its own deterministic **nested, stratified selection of training steps** at which the indirect pair is replaced with the direct pair. Selection uses only seed and training strata, **not model outputs**. Per-stratum counts differ by at most one for the 1%, 5%, 20% arms. Exposure slots at greater doses contain every slot exposed at lower doses.

| Arm | Direct `PA+AP` updates | Rate of **eligible pairs** | Approximate AP source fraction over all training sources |
|---|---:|---:|---:|
| `p000` | 0/6400 | 0% | 0% |
| `p001` | 64/6400 | 1% | 0.0625% |
| `p005` | 320/6400 | 5% | 0.3125% |
| `p020` | 1,280/6400 | 20% | 1.25% |
| `p100` | 6,400/6400 | 100% | 6.25% |

**Critical naming:** A “20% dose” means **20% of dose-pair training steps are converted**, not that 20% of all 16 training examples are AP. Each direct step contributes **one AP source** out of sixteen. Report this denominator explicitly in figures.

## Locked evaluation surfaces

- Use only the existing 100 **non-reserved development four-name sets**, never the reserved 95 name sets.
- Construct fresh, deterministic four-participant event orderings with no ordering reuse from old familiar development, EXP-109, or EXP-114 and no surface text overlap; separate exclusion audits precede training.
- Fix RNG seed `1172026`. Select **16 anchors per each of nine familiar modifier combinations**, 144 anchors total. Produce their AP, literal PA clause-swap, AA and PP presentations: **576 source presentations**; each has four standard role questions.
- Check parse/gold consistency, distinct names, lexical bags for AP and literal-PA, token-length constraints, vocabulary coverage, partition provenance, no L+I modifiers and exact evaluation digest.
- This is disjoint *within the same grammar and development name partition*; it is **not a new structural grammar or natural-language OOD dataset**. The original protected test remains closed.

## Frozen outcome and gate

For every model and each of the four evaluation conditions record: strict four-query `C_query`, per-query native four-name candidate-restricted correctness, wrong-role rate, wrong-event rate and correct-role logit margin. Report AP second-event mean correctness (second-agent and second-patient weighted equally), reciprocal PA first-event retention, and AA/PP controls. Record training losses at eight fixed checkpoints **without** intermediate scoring, adaptive learning-rate changes or early stopping.

The independent statistical unit is the **training seed** (two repeats averaged within each seed), **n=8**. Report two-sided 95% Student-t intervals (df=7) for seed-level averages and paired contrasts. Do not treat 16 MPS repetitions or 144 sentences as independent training runs.

**Primary prespecified directional test:** `p020` minus `p000` on AP second-event average query accuracy. Success: lower seed-level 95% interval bound >0.

**Secondary prespecified test (only interpreted if primary passes):** `p100` minus `p000` on the same measure, also lower bound >0.

The 1%, 5% exposure arms, nonlinearity/monotonicity of the full curve, all wrong-event/role rates, C_query, PA/AA/PP outcomes, and later mechanistic comparisons are **descriptive**. Even if all curves look monotonic, this study does not test or establish a globally optimal minimum training frequency. No selecting the “best” dose post hoc. No extra runs based on observed outcomes.

## Implementation and provenance checks

1. Regenerate original EXP-109 resources without touching reserved examples. Verify the EXP-109 and EXP-114 manifest hashes and reconstruct only their **already public development exclusions**.
2. In `--phase prepare`, create and audit new source *metadata* and every new seed's fixed schedule. Assert exact exposure counts, nested schedules, balanced modifier-frame allocation and token-bag equality. No model scoring or training.
3. `--phase smoke` uses a randomly initialized model, two training updates and **old already explored development sentences only**. It must not inspect new PHASE2 evaluation outcomes.
4. **Commit** the exact runner, launcher, protocol and generated manifest in one Git commit before any `--phase run`. The runner checks these committed bytes and their ancestry before training. Do not edit after freezing.
5. The launcher trains all 16 seed/repeat runs. Each trains all five arms from shared initialization and scores the frozen new evaluation only at the final step. Output reports and checkpoints must not overwrite existing files. Do not inspect intermediate reports or exclude weak models.
6. Summary requires 16/16 reports and verifies code/dataset/schedule/protocol hashes. **Only then** calculate all curves and the two fixed-order directional decisions.
7. Never commit `.pt` checkpoints; archive manifests, scripts, protocol and aggregate plus per-run JSON. Preserve the original EXP-109–116 archive unchanged.

## Scope, stop rule and future expansion

An observed improvement would show that **directly co-occurring construction evidence, relative to matched separate evidence**, improves generalization under this controlled synthetic training regime. It would not alone establish causal role-filler binding, isolate a semantic variable, or demonstrate out-of-grammar transfer. If the primary test fails, report failure and the full curve; do **not** change dose levels or pick a better-looking contrast on these outcomes.

Only after this study is fully reported should a **separate** structural-OOD study be designed, with deliberately different grammatical structures and new independent evaluation names/structures. Such a study requires its own generator and preflight to avoid confounding new grammar with untrained vocabulary or the closed L+I gate.
