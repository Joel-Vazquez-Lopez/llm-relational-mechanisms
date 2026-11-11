# ML-097 — prospective factorial source-transfer identification
**Status: PROTOCOL DESIGN DRAFT — NOT FROZEN, TRAINED, CALIBRATED OR EVALUATED. 2026-11-11.**

## 0. Decision boundary and relation to prior work
ML-090 has a confirmed paired-indexing bug (see `reports/ml090/CORRECTION-001.md`). This design never treats its original role-evidence numbers as evidence. ML-091–096 are discovery on previously examined ML-088 models and calibration cases. In particular, ML-093's norm matching sharply reduced an accuracy effect while retaining some role-gating margin differences; ML-094 source-end was mostly null; ML-096 half-strength removals gave little average rescue. Neither donor transfer nor ablation has established abstract role coding, unique native necessity, or mediation of the exposure benefit.

**Question:** Given a fixed receiving participant site, do norm-matched post-L1 donor-state edits change second-event answers according to the donor's grammatical role, lexical name, ordinal mention position, or an interaction with source/receiver context? Also: do these effects generalize over independently generated training seeds, held-out name pairs, source contexts, and questions?

**Single experiment:** a balanced 2 x 2 x 2 x 2 factorial of donor semantic role, donor lexical identity, donor mention ordinal and donor first-event context, crossing two fixed receiver participant sites. Repeat on active and passive second-event receivers. No other layer search.

## 1. Concrete contrast
Receiver: `While Aria helped Bram, Cleo was followed by Dax. In the second event, who was the agent?`
Gold: Dax; second-patient answer: Cleo; first-agent and first-patient names: Aria and Bram. (Illustrative only. These are NOT future critical dataset items.)

Patch the post-block-1 residual **at the receiver's source token Dax**; donor state is obtained before `<sep>` and without a question. Four donor cases with the **same donor name Dax**, same target verb and controlled first-clause context:

| Donor role of Dax | Dax ordinal in second clause | Donor second clause | Voice |
|---|---|---|---|
| agent | first | Dax followed Cleo | active |
| agent | second | Cleo was followed by Dax | passive |
| patient | first | Dax was followed by Cleo | passive |
| patient | second | Cleo followed Dax | active |

The first-event clause of each donor is a fixed, alternative, length-matched context that is the SAME across all four cells, not the receiver's unchanged first clause. This prevents the matched role/position passive cell from being a trivial identical-state transfer. Repeat with donor Dax replaced by an unused, in-vocabulary fifth name, with other second-event name Cleo fixed. Repeat with a second, preselected distinct first-event context. Repeat for the receiver's Cleo site. Do not choose donor texts or norm strengths based on observed effect.

For each factor vector `(role, name_match, ordinal, context)`, compute all five questions about the same receiver source sentence: first-agent, first-patient, second-agent, second-patient, and nonrelational `second-event first-mentioned name`. Apply identical source patch irrespective of the question.

## 2. Identifiability — limits that must not be hidden
The original grammar has only two second-clause voices:
- Active: agent first, patient second.
- Passive: patient first, agent second.

Thus **voice is deterministically role XOR mention ordinal** for a single participant. Even a balanced 2 x 2 role-by-ordinal factorial cannot independently estimate a role-by-position interaction against voice without a third syntax. Source-token **absolute index and prefix history** also differ across active/passive; patching post-L1 full states imports lexical, semantic, positional and general-context information simultaneously. Literal-name mismatch changes the donor identity representation and may cause off-vocabulary-of-candidates copying or OOD interference. A null under mismatched names does not refute a role feature.

The experiment can identify WHICH donor factors are *predictive of effects under this intervention*, not an abstract, voice-invariant representation or native necessity. The **smallest prospective additional control for that stronger claim** is a second, natively learned construction with independently variable agent/patient and order at fixed voice/length (e.g. counterbalanced overt agent/patient case markers supporting both permutations), plus calibrated native competence, matched position/prefix and the same patch test. Do not improvise untrained markers on critical examples; that would be OOD rather than an identification control. This is a later, separately frozen extension if needed.

## 3. New data, new models, and sealed evaluation
- New scratch-trained `ml080` 3-layer, width-96, 4-head causal Transformers, same 12,800-step AdamW framework, independent seeds `(9701,9703,9719,9721,9733,9739,9743,9749)`. Two paired exposure arms (`familiar`, `target`); **16 new checkpoints total**. Never reuse ML-088/080/084/085/086 checkpoints.
- New 16-name vocabulary disjoint from earlier ML-080/084/088 cohorts, fixed with a new K16 balanced matching partition (9 rounds training, 3 calibration, 3 sealed held-out; 72/24/24 disjoint participant pairs). Freeze exact names, matching seed `97097`, source code hashes, and each split ID before training. No earlier protected/ML-088 questions generated or inspected.
- Train arms on identical 56 base relational examples + 8 arm-specific relational exposure examples per step. Arm-specific 64 questions are repeatedly sampled as in ML-080. Add **8 equal retrieval examples per step in BOTH arms**, sampled with a separate fixed RNG. These train a literal first-mentioned-name question using existing vocabulary, allowing a meaningful nonrelational control. A retrieval example may expose the second-passive **surface form to both arms**, but NEVER its agent/patient query; therefore ML-097 tests effects of differential *role-query* experience, **not** completely withheld surface-form exposure. Do not describe the changed task as an exact ML-080 replication.
- Calibration: 2 fresh anchors per 24 calibration pairs = **48**, independent of all critical pairs. Critical: 4 fresh anchors per 24 held-out pairs = **96**. Predefine generator ordering with SHA256 of semantic factors (not of model responses); balance target direction, frames, verbs, distractor voice and source length as closely as feasible. If balance assertions fail, STOP and revise BEFORE freeze.
- Each anchor has a second-active and second-passive receiver; five question variants per sentence; other three names remain distinct. **Critical: 96 x 2 x 5 = 960 receiver prompts per model.** Calibration: 48 x 2 x 5 = 480. Do not write protected texts into version control; generate them only on a gated one-time critical run, with private salted commitments to selection rules.

## 4. Exact intervention
- Site: **post-L1 source participant residual, before L2**, once, at one recipient source participant token. Two recipient sites: second-event agent and second-event patient, separately. Never patch the answer/readout token. Native calculation uses unmodified output head.
- Donor factor cells per site: 2 donor roles x 2 lexical matches x 2 first/second mention positions x 2 first-event contexts = **16**. With two recipient sites: **32**. A shared first-clause substitution creates a nonidentical context, even when name, role and position match. Keep donor target verb, frame and co-participant constant within a factorial block; use the same predeclared donor context across matched factor cells.
- Controls per site: `self_identity` (strict zero); **four deterministic random unit-vector perturbations** scaled to the identical norm; with two sites that is 10 control conditions. Total **42 patch conditions + one native condition = 43**.
- For each receiver/site/context block compute `delta = donor_state - receiver_state`. Define `s = min( ||delta_j||_2 for all 16 donor cells )`. Scale all **nonzero** delta directions to exactly `s`; random controls also have norm `s`. If any delta norm < 1e-6, keep the cell but flag the block `non_identifiable_equalnorm`; do not artificially amplify zero or remove that block from accuracy/harm denominators. Identifiability gate on CALIBRATION only: >=95% of blocks must have all 16 norms >=1e-6 or halt before critical evaluation. Report raw norms, s and scaling ratios.
- Reuse exactly the same donor state and perturbation for all five question variants. Assert donor source index < `<sep>`, question invariance of donor source traces, finite logits, self-identical logits (max absolute error <1e-4), native trace parity (<1e-4), full-vocab output, and no effect from patching a source state **after final block** (<1e-6, preflight only).
- Donor-first-event contexts are independent, known tokens; record sentence length and absolute source index. Hold out one entire first-event context family from calibration parameter fitting and evaluate it without tuning in the critical run.
- Do not use original ML-093 equal norm code blindly: its original matched/reversed control has different factor structure. Implement and test the common-norm calculation anew.

## 5. Outcome definitions and baselines (all frozen before critical)
Let `A,P` be true receiver second-event agent and patient names. For a source sentence, define question-conditioned role evidence:
```
G(z) = .5 * [(z(second-agent,A) - z(second-agent,P))
            -(z(second-patient,A) - z(second-patient,P))]
D = G(z_patched) - G(z_native)
```
Index **each question row explicitly**, not via cross-example tensor indexing. `D` is continuous in raw logits. For each model normalize by a fixed positive scale: median absolute native agent-vs-patient contrast on the **calibration** second-active set, floored at 1.0; save scale before critical evaluation. Keep both raw and normalized margins.

For every condition also save: unrestricted argmax and full-vocabulary correctness, exact four-name candidate ranks and rankings, native predicted answer vs patched predicted answer, patch-target name and donor name token ranks, receiver gold-vs-best-other margin, true four-choice accuracy, confusion (wrong event vs wrong role), native-error-to-correct rescue count, native-correct-to-error harm count, wrong-to-different-wrong transitions, source edit norm, output-logit L2 and KL divergence, and a **first-mentioned-name retrieval-control** margin and accuracy. Do not relabel candidate-only wins as unrestricted output success. Report query-by-query, event-by-event and active/passive strata, not only an average.

Before opening the critical set, on *calibration only*, fit exactly these predeclared ridge models (fixed penalty `lambda=1`, common feature coding and calibrator):
- `H_R`: donor grammatical-role match with receiver site; receiver question role.
- `H_N`: literal donor-name match/copy and donor-name logit prior; no grammatical-role feature.
- `H_P`: donor ordinal / receiver ordinal / absolute index and source length; no role.
- `H_B`: native output preferences and margins plus edit norm, donor-vs-receiver logit prior; no donor role.
- `H_G`: generic matched-random perturbation mean / output-scale model; no donor factors.
- `H_I`: same main effects plus all predeclared pairwise role x name, role x position, role x source context and role x receiver voice interactions (ridge; no feature selection).

Fit separately by seed/arm on calibration instances only; no model selection, coefficient tuning or thresholds after viewing critical logits. Freeze a file containing predictions for **every critical condition and outcome**, including source-factor covariates, native preference baselines and all model coefficients; commit its SHA256 before any critical logits are evaluated. Prediction generation may use prompt features and sealed item IDs but not critical model inference. If outcome-dependent information is required for prediction, predict **patch minus native using calibration native measures only**, and treat critical-native baseline analyses as a separate descriptive covariate. Do not fit a native-preference baseline using critical native logits after examining them: score its prediction from the calibration-fitted model with fixed predeclared per-item prior proxies.

**Primary metric:** per-seed squared-error improvement on normalized `D` of each hypothesis model over the strongest of `H_B,H_G`, averaged across all frozen critical factor cells; report `H_R` versus `H_N,H_P,H_I` and `H_I` versus the strongest additive rival. At most one preregistered selection statement: the winning explanation must lower critical MSE >=5% relative to **every non-nested competing simple model** AND show a positive improvement in >=6/8 paired seed replicates with paired-seed bootstrap 95% interval excluding zero. Report all losses without declaring a winner if multiple accounts satisfy criteria. Do not use per-prompt p-values as independent replication.

Supplement: per-site mean contrast of role-aligned versus reversed role donors at fixed donor name, mention ordinal and first-event context; role x position, role x lexical, role x context interaction estimates; difference-in-differences of target-minus-familiar exposure arms; and predictive relationships to native second-passive generalization. Use cluster/seed uncertainty, not sentence-level IID intervals.

## 6. What outcomes would mean
| Account | Distinctive prospective pattern | Disconfirming or insufficient outcome |
|---|---|---|
| **Grammatical-role-sensitive transfer** | A role-matching donor systematically shifts `G` in the preregistered direction across both mention ordinals, both lexical conditions, both source contexts and receiver voices; `H_R` predicts new seeds/cases better than position/name/native/generic rivals; real answer changes follow | Effect disappears after norm matching, is confined to same names or one voice, fails to outperform strong rivals, or only changes candidate-restricted scores. At best role-*sensitive under intervention*, not abstract representation |
| **Literal-name / copying** | Donor-name identity strongly predicts full-vocab rank and unrestricted outputs; same-name and changed-name controls switch effects while role and ordinal are fixed; `H_N` dominates | No copying and `H_N` fails on held-out patterns |
| **Mention-position** | First/second donor ordinal and absolute-index model predicts transferred effects independent of donor semantic role/name, with retrieval-control changes tracking source position; `H_P` dominates | Ordinal predictions reverse between active/passive when role held fixed; `H_P` does not generalize |
| **Contextual interaction / multi-cue routing** | Strong, reproducible role x name/position/context or receiver-voice interaction; `H_I` yields clearly superior out-of-sample predictions, beyond its extra flexibility | Only unstable seed-specific interactions, coefficient flips, failures on held-out contexts, or simple native/norm baselines matching it |
| **Generic perturbation** | Equal-norm random changes output/margins/harm as much as structured donors; `H_G/H_B` matches full models | A replicable, selective and predictive directional effect of structured donors exceeds random and native-prior controls |
| **Non-identification** | No competent native target behavior, no norm-identifiable donor blocks, mostly no answer changes, calibration-only correlation, or violated negative controls | Report as inconclusive; do NOT turn a null into evidence against role representations |

**Falsifier of the favored role-transfer explanation:** role-aligned transfer fails the above prediction and control criteria even where native target and retrieval tasks are competent, and position/name or generic perturbation baselines explain the held-out effects. Conversely, observed role sensitivity alone is NOT proof of a native role circuit.

## 7. Training-experience link
Report within each paired seed: target-minus-familiar native second-passive accuracy, native corrected role contrast, which donor model best predicts outcomes, and the role-vs-nonrole selective intervention contrast. A causal claim that the **training exposure creates or mediates** the mechanism requires a later experiment randomizing/altering that intermediate computation without answer donation; cross-arm correlations or transplants here are insufficient. All synthetic results remain about this small architecture and grammar. Pretrained LLM transfer is a separate, untested question.

## 8. Exact implementation contract, phases and hard budget
Implement `scripts/ml097_factorial_source_transfer.py` with the following **fixed, reviewed phases**:
1. `--phase plan`: freeze model IDs, new names, matching seed, training recipe, lexical donor selection, exact 48/96 semantic anchor IDs, factor ordering, all methods/thresholds, code/dependency SHA256 and resource budget to `reports/ml097/frozen_protocol.json`. No critical sentences, checkpoint inference or results.
2. `--phase preflight`: independently unit-test rowwise contrast, source token indexing, donor factorial orthogonality, full-factor balance, no answer-token leakage, self/null, random equal norms, L3 no-effect, and isolated retrieval parsing on untrained models only.
3. `--phase train --until 3200`, then `--until 12800`: train both arms/8 seeds deterministically; save checkpoints and complete native calibration metrics. No protected data generation.
4. `--phase calibrate`: check >=6/8 per arm reaching >=95% native familiar-relational calibration accuracy AND >=90% both-role competence, with >=90% first-mentioned retrieval accuracy in BOTH arms; test >=95% nondegenerate calibration donor blocks. If any global gate fails, STOP and preserve negative results. Fit five simple + interaction predictors on 48 calibration anchors, make frozen per-condition critical prediction manifest; commit and record hash before evaluation.
5. `--phase evaluate --confirm-protected --predictions-sha256 <hash>`: refuse unless protocol, all checkpoint IDs, competence/calibration, saved predictions and SHA digests match, no prior protected-open lock/report exists. Atomically mark one-time open, construct 96 critical anchors once, run native then all interventions and diagnostics, save to `reports/ml097/protected_evaluation.json`, never overwrite. Report all seeds; no new tuning.
6. `--phase summarize`: append a separate interpretation document; never alter frozen protocol, predictions or raw report.

**Exact compute workload:** 8 paired seeds x 2 arms x 12,800 optimizer steps = **204,800 training steps**. Each step contains exactly 56 base + 8 exposure + 8 retrieval examples = **72**, thus **14,745,600 training prompt examples**, exclusive of fixed milestone evaluation. Sixteen models at two save milestones (3,200, 12,800). Critical inference: 16 models x 960 receiver questions x 43 native/patch conditions = **660,480 forward prompt evaluations**, plus at most 16 x (96 anchors x 2 voices x 32 donor source sentences) = **98,304 donor source traces**. Calibration: 16 x (48 x 2 x 5) x 43 = **330,240 prompt evaluations**, plus at most **49,152 donor source traces**. **Planned non-training evaluations: 990,720 question forwards + 147,456 donor traces**; fixed competence-gate and synthetic preflight costs are extra and must be logged.

**Resource ceiling**: one accelerator or CPU worker at a time, FP32, deterministic Torch seeds, training batch size 72, inference batch size <=64 (or smaller if sequence shape requires), max 4 GiB accelerator-memory allocation, max 4 GiB experiment artifacts, and max **96 aggregate device-hours** for the cohort. Benchmark 200 *untrained/synthetic* train steps first; if projected training exceeds 96 device-hours, STOP and amend/reapprove budget before any new model-training or protected evaluation. The exact wall-clock duration cannot be known without benchmarking the user's device; do not present an invented runtime estimate. Record torch/Python, GPU/MPS/CPU, commit SHA and numerical tolerance.

**Current implementation status:** this is a complete frozen-target execution specification, **not** a claim that the ML-097 runner or future prediction files already exist. Implementation and review of the runner, factor balance assertions, resource benchmark, and approval of the final hash are mandatory *before* any data-generating run. No critical experiment has been executed.

## 9. Integrity / stop rules
Any ambiguous donor mapping, norm clipping from zero, identity non-parity, position/name alias not logged, retrieval control incompetence, future seed reuse, calibration/critical leakage, code/hash mismatch, or protected re-open request STOPs evaluation without ad-hoc fixes. Preserve every failure, including ML-083, ML-084, ML-087, ML-094, ML-096, the ML-090 correction, and all no-effect arms. No transfer of these claims to pretrained LLMs without separate prospective tests.