# ROLE-DIVERSITY-COMPLEX-v0 — proposed 1/2/4/8 training study

**Historical source-only design note.** The full prospective runner, fixed 16-seed schedule, protocol hierarchy, source/evaluation preflight, and CPU random-weight smoke are now implemented and [validated](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/actions/runs/45285668723). This historical document predates that implementation. For the current procedure, use [ROLE_DIVERSITY_COMPLEX_V0_PROTOCOL.md](ROLE_DIVERSITY_COMPLEX_V0_PROTOCOL.md) and [ROLE_DIVERSITY_COMPLEX_V0_RUN_FULL.md](ROLE_DIVERSITY_COMPLEX_V0_RUN_FULL.md). The model study has NOT been trained and is not frozen until both local manifests are committed.

**Status: source-only design validated, NOT YET A FROZEN TRAINING PROTOCOL.** Do not run any new-seed models, use heldout target outcomes to select conditions, or mutate prior scientific studies. A new prospective seed list, exact training code, per-seed quartet/cell/role schedules, full vocabulary, model checkpoint guard, scoring implementation and study manifest must be written, audited, and Git-frozen BEFORE the first training run.

## Question

Under a **combined neutral-prefix + named-adjunct** training distribution, is it necessary to encounter multiple different semantic role arrangements for the SAME four participant names to generalize to never-trained role permutations?

The prior 12-seed TRAIN-SURFACE-FACTORIAL-v0 found severe combined-condition seed bimodality on heldout role permutations, despite nearly perfect original-core accuracy. The follow-up cross-surface diagnostic found almost no rescue from matching the input surface; this study varies **training role diversity** while retaining that complex source distribution. The hypotheses are motivated by earlier results and must be tested on FRESH independent seeds, not retroactively proven by the old cohort.

## Four nested treatment arms

All use the exact 240 originally TRAIN-derived four-name quartets, each quartet's same four participant names and two event verbs, and two familiar voice templates, AP and PA. Training source variation is crossed across the SAME 5 neutral prefixes (including no prefix) and 9 bare/L/I paired modifier cells (including bare/bare, excluding within-event LI). Across arms, model initialization, 6,400 optimization updates, 8 selected quartets per update, two voices and four supervised labels per source MUST be identical within each new seed.

Let \`(a,b,c,d)\` be a quartet's original four role names, in first-agent, first-patient, second-agent, second-patient order. The previously checked \`all_roles()\` defines 8 allowed *dihedral* role permutations and another disjoint 16 heldout permutations. Arms use strictly **nested** training assignments:

| Treatment | Allowed training assignments | What changes? |
|---|---|---|
| \`roles_1\` | first original assignment only | Fixed name-to-event-role mapping |
| \`roles_2\` | first original + first 1-step cyclic rotation | Every name sees TWO different role slots |
| \`roles_4\` | first four cyclic rotations | Each name reaches all four role slots once in the eligible set |
| \`roles_8\` | four rotations plus four reversed rotations | Each name reaches each of the four slots twice in the eligible set |

**All arms leave exactly the other 16 permutations per TRAIN quartet unseen**, making evaluation common. All 24 role assignments of the further 60 previously TRAIN-derived withheld quartets are also withheld from these new models. Original DEV, reserved/protected and within-event LI are excluded. Both AP/PA voice constructions are shared across all arms. We do not assume success just because some permutations were seen: models must transfer to those **never trained**.

## Matched design versus intended differences

All arms should have:
- identical step/quartet/voice/45-cell surface schedule and target count per optimizer update;
- identical architecture, model initialization within each seed, optimizer, max input positions, source token lengths and non-name token bags for a chosen quartet/voice/frame/modifier cell;
- new, distinct seeds from the older 139xx and 149xx scientific cohorts; proposed 16 seeds ×4 arms =64 models, subject to pre-run power feasibility and protocol freeze;
- no seed or checkpoint filtering based on accuracy; no early stopping; one final checkpoint per arm.

**Intended treatment differences** include the number of distinct training role arrangements, the lexical names in corresponding source positions and distractor positions, target label frequency at each query slot, and number of unique training examples. These are NOT controlled away; they are the mechanism-level learning-distribution manipulation. Training exposure to each allowed role permutation must be audited within each quartet across the exact 6,400-step schedule, with at most one exposure count imbalance. Also audit distribution of role assignment versus 45 surface cells, since their interaction could otherwise confound conclusions. No full training implementation yet exists.

## Proposed prospective hypotheses (not frozen)

A useful hierarchy to freeze after source/schedule checks is:

- **Primary**: seed-paired \`roles_8 − roles_1\` in native strict four-question joint correctness (\`C_query\`) on all **16 never-trained roles per TRAIN quartet**, using a preassigned *rich* evaluation surface (nonempty prefix plus one permitted nonbare paired named modifier) with common AP+PA voices. Proposed positive-lower-bound two-sided 95% seed t interval and ≥+10 percentage points mean practical effect, plus a predeclared across-seed sign gate.
- **Gated secondary**: \`roles_2 − roles_1\` on exactly the same frozen rich test, only confirmatory after successful primary. The \`roles_2\` contrast tests whether exposure to **one additional assignment repeatedly** improves transfer. It does **not** test one-shot learning.
- **Exploratory**: \`roles_4\` dose, dose-response shape, per-seed bimodality, comparisons on identical bare sources and on completely withheld four-name quartets, accuracy on rich original training cores and event-pair errors. Potential 1-shot variant belongs to a *different* study.

All 16 heldout permutations remain fixed regardless of treatment, and both rich/bare evaluation source renderings are independently parsed and gold-checked *before* any new model score. Do not adjust threshold/direction/seed list after looking at new data; report negative findings honestly.

## Source-only validation completed

The companion script \`scripts/role_diversity_complex_v0_preflight.py\` passed an automated GitHub Actions source-only audit. It checked all 8 training-eligible assignments on bare input for all 240 TRAIN quartets (3,840 voice-source items); exhaustive 8×45×2 source variants on 24 deterministically sampled quartets (17,280 items); 7,680 unseen-role bare and 7,680 unseen-role rich source items; 2,880 heldout-quartet bare and 2,880 rich source items. Each source was parsed independently and stayed within the 48-token limit. Different role assignments have identical length and non-name token bags for each same quartet/voice/surface cell. No model training or inference occurred. The design audit is not a frozen scientific training manifest.

## Immediate gate

**First** complete ROLE-ANCHOR-PROBE-v0 on existing checkpoints. Its exact-original-tuple and per-quartet mode statistics can help distinguish genuine old-core anchoring from alternative rigid shortcuts. **Then** finish and freeze the new diversity training implementation and analysis hierarchy, including exposure balance. Do not conflate the anchor audit with independent confirmation or use its outcomes to select post-hoc test thresholds.
