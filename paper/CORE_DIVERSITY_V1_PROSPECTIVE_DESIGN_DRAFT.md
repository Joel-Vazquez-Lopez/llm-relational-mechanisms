# CORE-DIVERSITY-v1 — candidate prospective training intervention

**Status: DESIGN DRAFT ONLY. NOT FROZEN. NO TRAINING OR CHECKPOINT SCORING AUTHORISED BY THIS FILE.**
Written after observing the exploratory CORE-TRANSFER-v0 category means; a design selected after seeing the old results cannot be presented as a replication of the old analysis. This proposal becomes a new study only after the existing read-only reports (including bare/bare and seed-level outcomes) are independently audited and a source-only protocol, experiment code, new evaluation fixtures and hashes are committed *before the first new model inference*.

## Decision motivating the new study

POSITION-CB-v0 used one underlying training dose core per original train name quartet and many repeated surface variants. The completed read-only train-pool audit found ~95% pooled strict joint accuracy on selected and never-selected variants from those same cores but ~37% on the original familiar development distribution.

The preliminary CORE-TRANSFER-v0 console summary supplied by the experiment operator (pending remote raw JSON reconciliation) shows that simply flipping roles, swapping event order, or repartitioning the SAME four train names drops familiar-voice joint accuracy to about 32–37%, close to familiar development, versus 92–98% on unchanged cores. The three novel-core categories have similar means; it is therefore **not established** that one class of semantic transformation is uniquely responsible. Our working causal hypothesis is a dependence on repeatedly sampled, source-core-specific participant/role configurations.

## Falsifiable prospective hypothesis

With exactly matched four-name-set identities, model architecture, optimizer, training-example count, voice-pair exposure, per-example lexical bags and source lengths, **greater diversity of underlying event-role assignments per training quartet** causes better accuracy on role assignments never used for training. An additional hypothesized benefit is generalization to completely unseen name quartets drawn from the same 12-name vocabulary.

The claim is **behavioral and training-distribution causal**, not a claim that a specific head or circuit encodes roles.

## Data split — do not access original protected partitions

Reconstruct the original 300 TRAIN quartet identifiers **only** (not original DEV 100, not original reserved/protected 95). Deterministically split these 300 old TRAIN identifiers into:

- **240 train-only quartets**, appearing in both candidate training arms;
- **60 entirely withheld quartets**, appearing in neither arm, for a *new, prospective, within-grammar quartet holdout*.

Use a fixed new random seed for the 240/60 split and commit identifier hashes before inference. The test name quartets remain lexically familiar at the individual-name level but are unseen as joint name sets **for the newly scratch-trained models**. The fact that prior old models saw these identifiers must be disclosed; it does not mean the new randomly initialized models were trained on them.

For each quartet define a unique anchor ordered role tuple (a,b,c,d) and two distinct verb identities, fixed across arms and across the role-permutation conditions. Use the bare/bare modifier for the **primary study** to guarantee exact token-multiset equality under every participant permutation at a fixed voice. Do not use original L+I material or any of the 95 protected name sets.

## The intervention

For each quartet, define eight distinct *training-eligible* role assignments using the dihedral permutations of (a,b,c,d): four cyclic rotations and four cyclic rotations of the reversal. This eight-element set has an important exact property: each participant occupies each of the four gold role positions **twice** across the eight assignments. The remaining **16 of 24** permutations are reserved as source-core-disjoint evaluation variants.

Two training treatments:

1. **Fixed-core:** always use the original (a,b,c,d) ordering for the chosen quartet, with the same number of source exposures per optimizer step.
2. **Core-diverse:** cycle through the eight training-eligible permutations according to a source-indexed, seed-deterministic schedule; every quartet should receive close to uniform allocation across its eight assignments, and no heldout permutation is ever used.

Hold constant, as exact assertions whenever possible: same quartet per step, same four participant names, same pair of verbs, same per-source token bag, source length, query types, gold-answer multiset, voice-pair presentations, optimizer/architecture/learning-rate schedule, and total optimizer steps. Changing **which participant carries which role within an event** is the intended treatment. Name-specific *position* and role frequencies are therefore necessarily changed by the intervention; these are candidate mediating mechanisms, not controlled-away confounders.

To avoid choosing a favorable voice regime after seeing outcomes, cross the treatment with BOTH original exposure pairings:

| Training-voice pairing | Fixed-core | Core-diverse |
|---|---|---|
| Same-voice: AA + PP for each source core | Arm A | Arm B |
| Mixed-voice: AP + PA for each source core | Arm C | Arm D |

Both regimes already balance first/second-event active/passive voice marginals. All four conditions use paired quartet schedules and paired model initializations within each seed. Compare A vs B and C vs D on **their own familiar voice pairings**, and average both matched causal contrasts for the primary estimand. Unseen voice pairings are a distinct secondary/exploratory question, not part of primary within-core transfer.

## Evaluation — identical items scored in treatment/control

1. **Primary test:** all 16 role permutations **never selected for training** on the 240 train quartets, scored in each regime's familiar voice combinations. Average strict four-query joint correctness first across heldout variants/voice surfaces, then paired differences between core-diverse and fixed-core within seed and regime.
2. **Gated secondary:** fixed, precomputed role permutations on the 60 completely withheld four-name quartets, with identical voice combinations and source/modifier controls. Test treatment-control difference only if the primary passes.
3. Descriptive: strict joint accuracy across individual event-role transformations, single-query four-name accuracy, full-vocabulary accuracy, paired-event error categories (correct/reversed/collapsed/wrong-event), and the unseen voice-combination response.
4. Baselines: predeclared simple participant/name priors and core-slot/positional heuristics, with no access to heldout model outcomes or label-derived oracles inaccessible to the Transformer.

**No model selection on new test sets.** Train-only or explicitly designated pilot data can be used to choose an optimizer setting **before freeze**; no test or protected evaluation may inform training duration, seed choice, hyperparameters or the final outcome definition.

## Seeds, inference and stopping

- Candidate scale: **16 fresh independent training seeds**, **one MPS run per seed/arm**, four arms = 64 trained states. This improves inferential units over eight seeds × two numerical repeats at a comparable total architecture. Reassess compute feasibility using **training-only** profiling before freeze, not downstream outcomes.
- Each seed draws the same quartets and voices for all four arms; initial weights identical across arms within seed.
- Prespecify a fixed number of updates, full-vocabulary token supervision, no checkpoint selection, no competence filtering, no early stopping based on evaluation.
- Primary effect: seed-paired mean `core_diverse - fixed_core` strict `C_query`, averaged across both training-voice regimes on the 240-quartet heldout-permutation set. Report absolute effect, paired seed-level two-sided 95% t confidence interval, all 16 seeds, and number with positive effect. Do not alter this after results.
- **Proposed success rule to freeze after feasibility checks:** two-sided seed-level lower 95% CI > 0, positive point estimate >= +10 percentage points, and at least 12/16 positive independent seeds. This is a design proposal, not yet registered.
- Only if the primary passes, test the gated 60-quartet new-name-set comparison using its separately prespecified interval criterion.
- Regardless of result, publish failures, all seed results and code. Do not rescue a failed test with retrospective competent-seed selection.

## Validation before first prospective run

Implement and test source-only generator and independent parser on all 24 role permutations. Verify train/evaluation permutation disjointness, train/withheld quartet disjointness, and no overlap with old protected/reserved sets. Verify exact token-bag and length equality for same quartet/permutation/voice between arms (bare/bare). Check model input lengths, gold answers, role-position counterbalancing within treatment, deterministic seed schedules and paired initialization. Freeze a manifest with hashes of source fixtures, hashes of all scripts/dependencies, and the primary/secondary analysis logic **before scoring any study model**.

The result could be a strong paper if increased semantic-core diversity prospectively improves independently withheld configurations, including separate quartet combinations, with reliable effects across new seeds. If it fails, report that and do not claim an identified role-binding mechanism. A different grammatical family and learned-shortcut baseline remain necessary before making broad claims about Transformers in natural language.

## Immediate dependencies

1. Archive all 16 existing CORE-TRANSFER-v0 JSON checkpoint reports and inspect the **bare/bare** control and per-seed breakdown; no raw reports were in the connected GitHub branch when this proposal was drafted.
2. Review this draft, particularly the 240/60 repartitioning, dihedral training assignment design, all four voice conditions, and compute budget.
3. Only then implement/freeze a new study with a new manifest and fresh-seed script. **Do not launch training based solely on this draft.**
