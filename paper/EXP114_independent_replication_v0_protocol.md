# EXP-114 — Locked independent-seed replication of positional and early-update intervention effects

**Status:** Prospective, fixed analysis plan for **new scratch-trained models and disjoint within-family development sentences**. The plan must be committed alongside the prepared manifest and exact runner **before any EXP-114 checkpoint is trained or scored**. It is a targeted replication of findings *selected* from the exploratory EXP-109–113 sequence, not external preregistration and **not** proof of a unique semantic-role circuit.

## 1. Scientific motivation and frozen predictions

EXP-109 observed a replicated AP versus literal PA generalization gap under PA-only training. EXP-110–111 described reversal-like second-event role failures and final role-logit differences. EXP-112 used source-clause activation replacement. EXP-113 separated explicit positional codes from first-block residual updates and observed on previously examined sources:

- PA-only AP second-event accuracy increased by approximately **+19.8 percentage points** under matched passive-clause *input positional-code* replacement.
- On PA-only AP, transferring the learned first-block update changed accuracy by roughly **+12.8 pp**, whereas adding the donor explicit position code *after Block 1* changed accuracy by roughly **+0.1 pp**.
- Reversing a first-block update transplant into the familiar PA passive first event reduced that event's query accuracy by approximately **−38.2 pp**.
- Exploratory controls identified strong dependence on participant-name tokens, but aligned versus reversed position replacement was not clearly distinguishable; wrong-event responses sometimes increased substantially.

EXP-114 freezes these observations as a testable prediction set. It does **not** infer that transferred Block-1 updates are position-free: the donor update depends on position, prior context, self-attention, and MLP computation. Accuracy changes may reflect event-selection changes as well as role assignment.

### Hierarchically ordered tests

For each independent training **seed**, average its two numerical MPS repetitions; within each repetition and intervention, average native correctness across **all 144 newly generated AP anchors**, and the **second-agent and second-patient questions equally**. Use eight seed-level estimates (n=8), Student t with seven degrees of freedom, two-sided 95% intervals. No outcome-driven thresholding, arm filtering, competence gating, checkpoint selection, or optional stopping.

1. **Primary:** In the `pa_only` arm, `input_passive_all` causes a **positive** change in AP *second-event individual-question* native four-candidate accuracy relative to the identical unpatched model on the same sources. Criterion: lower 95% CI bound strictly greater than zero.
2. **Secondary 1, gated on primary:** In `pa_only`, the AP second-event native accuracy improvement from `late_update_passive` is **greater** than that from `late_position_passive`. Criterion: lower 95% CI bound strictly greater than zero for the paired difference.
3. **Secondary 2, gated on passing the first two:** In `pa_only`, `late_update_passive` transferred from AP into PA causes a **negative** change in familiar PA *first-event* individual-question native accuracy. Criterion: upper 95% CI bound strictly less than zero.

This fixed-order procedure avoids promoting unsuccessful later tests as independent confirmations. Both arm-level results from `indirect_pp_aa` and `direct_pa_ap` are retained **descriptively**, never filtered and never redefined as primary. The nominal intervals rely on training-seed variation; they cannot establish mechanism uniqueness, out-of-grammar generalization, or future-model reliability.

## 2. Model and training hold constant

Use **8 NEW training seeds**: `11401, 11403, 11407, 11409, 11413, 11419, 11421, 11427`; **two MPS numerical repeats** each; **3 unchanged arms** per run (`pa_only`, `indirect_pp_aa`, `direct_pa_ap`). This produces **16 full training runs and 48 model states**, each at the **fixed 6,400-update endpoint**. Reuse EXP-109/106 architecture, optimizer, training sources, step-level arm exposure matching and training schedule algorithm, changing only training seeds. No warm start, model selection, architecture sweep, or further training of old checkpoints.

Run source schedule SHA-256 audits for all new seeds before training. Persist a frozen manifest with dependency hashes, evaluation digest and schedule hashes. Store checkpoints under `results/exp114/independent_replication_v0/` locally; never commit `.pt` files. Reproducibility is numerical rather than bitwise on MPS: **both** repeats are included regardless of competence.

## 3. New evaluation, with strict old-data exclusions

Use the **existing 100 non-reserved development name sets** (no reserved 95 name sets). Within those sets, construct **new agent/patient orderings** excluded from:

- the *full* previously generated two-core-per-name-set old development core set;
- EXP-109's already evaluated AP and literal-PA orderings;
- all old explored development source surfaces and all six EXP-109 evaluated condition surfaces.

Use deterministic seed `1142026` to generate a fresh pool, then sample exactly **16 anchors from each of 9 legal modifier combinations** (144 AP anchors), with their 144 **literal PA** clause-swap counterparts, preserving lexical token multiset and the corresponding relabeled first/second gold answers. Validate independent parses, four distinct participants, token length, source disjointness and all within-clause lexical mappings. Audit and freeze the evaluation digest; **never write the fresh sentence strings to the reports**.

**Important scope limitation:** new source *sentences* and training *seeds*, but not a new synthetic grammar, vocabulary, four-name partition, or real natural-language dataset. The protected/reserved/L+I evaluation remains **closed**, and its exemplars are never constructed or inspected.

## 4. Frozen interventions

Run within each trained model and on the same AP/PA paired source examples. Retain query/answer token states; modify only matching upstream source-clause token states.

| Mode | Causal operation | Role |
|---|---|---|
| `input_passive_all` | Replace PA→AP (or AP→PA) **absolute position code** for matching passive-clause tokens at input, then recompute Blocks 1–3 | Primary |
| `late_update_passive` | At Block 1, use donor update `U_d=h_d^(1)−[E(t_d)+P(d)]` while retaining recipient `E(t_r)+P(r)` | Key secondary |
| `late_position_passive` | At Block 1, replace only explicit `P(r)` with `P(d)`, retaining recipient precomputed update | Key secondary comparator |
| `late_full_passive` | Transfer both donor position code and learned update, reproducing EXP-112's full Block-1 patch | Algebraic check/descriptive |
| `input_active_all` | Matched active-clause input positional-code transfer | Context/control |
| `input_passive_reverse` | Reverse within-clause **position-code** correspondence, preserving original lexical token embeddings | Alignment/control |
| `input_passive_core_names` | Replace input position codes only at two participant-name tokens | Post-hoc-inspired specificity control |
| `input_passive_voice_cues` | Replace input position codes only at *was*, verb and *by* | Post-hoc-inspired specificity control |

Intervention selection, direction, question categories and depth are all frozen now. The Block-1 input and update are related by

`h1 = E(token) + P(absolute index) + U1`.

These are **interventions**, not observational feature importances. `U1` can encode position and prior context; all patching may yield off-manifold states. Aligning same lexical tokens does not imply a semantic-role-only change, and an improvement on AP does not alone rule out wrong-event substitutions.

## 5. All recorded diagnostics and control interpretation

For every arm, AP/PA recipient direction and the eight fixed interventions, record: native four-name per-query accuracy, strict four-query `C_query`, correct-versus-opposite-role logit margin, wrong-role rate and wrong-event rate; each patched-minus-native query delta. Save checkpoint/training schedule/protocol/source digests. No per-example sentences, hidden activations, logits or names in JSON.

Descriptive controls include: input passive versus input active, aligned versus reverse position codes, participant-name versus grammatical-cue token groups, AP first-event collateral effects, PA first-event reciprocal effects, and wrong-event changes for each intervention. Control effects need **not** be null for the primary prediction to pass; if they show comparably large non-specific disturbance, mechanistic interpretation is weakened and must be reported.

Any post-hoc proposed refinement after seeing EXP-114 results is clearly **EXP-115 exploratory** or a separate independently frozen study, never a retrospective change to EXP-114.

## 6. Protocol-freeze and reporting workflow

1. Install the EXP-114 Python runner, shell launcher, and this protocol into the research worktree.
2. `--phase prepare`: construct and audit only new development source *metadata* and deterministic training schedules; **do not train or score models**. This writes `reports/exp114/independent_replication_v0/manifest.json`.
3. `--phase smoke --device mps`: only a random model and already explored legacy development examples are used for intervention smoke tests; two training updates on standard training sources. The new 144 evaluation pairs are not scored.
4. **Commit the exact runner, launcher, protocol and manifest in a single Git commit before any `--phase run`.** The runner verifies those exact bytes in a Git ancestor. Include the ignored manifest with `git add -f`. Subsequent source or manifest mutation causes a hard failure; never overwrite run reports.
5. Execute all 16 fixed `(seed, repeat)` runs, each training three arms and scoring only the 144 new AP/PA pairs. Reports are sealed during training: print only checkpoint loss and save final diagnostics. **Do not inspect partial outcomes or decide to stop/modify based on them.**
6. `--phase summary` aggregates **only after all 16 reports** are present and verified. It computes the primary and ordered secondary decisions plus full controls with eight seed-level means/intervals, reports all results and never filters by competence.
7. Commit results and aggregate JSON after checking no secrets, reserved data or checkpoint tensors are staged. No GitHub push is required for scientific inference, but pushing the frozen plan *before* training makes chronological provenance easier to verify.

## 7. Claims permitted and not permitted

If the primary prediction passes, supported: **independent training seeds and disjoint within-family source examples reproduce sensitivity of AP second-event accuracy to position-code editing**. If both ordered secondary predictions pass, supported: **for these models and synthetic contexts, Block-1 learned updates have a larger impact than late explicit position editing and reverse transfer damages familiar PA performance**.

Not supported even after success: a unique semantic-role circuit, abstract causal mediator, a position-only heuristic across architectures, natural-language generalization, immunity to event-selection confounds, or robustness to a new grammar/name-partition. A failed test falsifies its prespecified intervention prediction for this setup, not the existence of all possible position-sensitive representations.
