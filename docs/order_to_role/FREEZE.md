# FREEZE — order-to-role, test set 0 (26 Nov 2026)

Status: FROZEN by the commit that adds this file (together with `scripts/otr_freeze_score.py` and
`scripts/slurm/otr_test0_submit.sh`). Frozen from results already in hand (option B, 25 Nov), self-reviewed
by Joel and Claude and reviewed adversarially by Codex against the code (24 Nov; 11 points addressed).
There is no supervisor sign-off; the public GitHub commit timestamp is the record. No `--split test0`
command may run before this commit. Nothing here was chosen on test data. After the commit, sections
2–6 do not change; anything else is labelled exploratory.

## 0. Prior access to held-out data (as of 26 Nov)
- No result file with a test split or status `TEST_SET_OPENED` exists in the results repository
  (checked 24 and 26 Nov: 0 files). The older OTR-01 runner has a `--split test --test-set 0` option; it was
  never run on a test split.
- Disclosure: dev OTR-03 runs drew donor name pairs from all name pairs, including pairs assigned to
  test sets. Donors are counterbalanced (each donor name in each role) and no donor-level result was
  inspected. From patch 0066 on, test runs draw donors only from the run's own split and abort on any
  leak; dev runs keep the original draw so archived dev results reproduce exactly.
- The timestamp shows only that this file precedes the first `--split test0` command.

## 1. Scope
- Models. Inclusion rule on dev (instructed extractor score): ordinary questions >= .80 and working
  question >= .90. Mechanism models: Gemma 4 31B-it, Qwen3.8-27B (thinking off), phi-4,
  OLMo-3.1-32B-Instruct, Granite-4.2-30b (thinking off). gpt-oss-20b fails the gate (about .76 ordinary,
  .88 working on dev). Its C1 numbers are reported descriptively and are not counted in any pass count.
- Prompting: chat template; instruction "Reply with one name only." in every run.
- Model revisions: the Hugging Face snapshot cached on Pelle (one snapshot per model; jobs run with
  `HF_HUB_OFFLINE=1`, so dev and test runs use the same weights), listed in section 4.
  Runtime: bfloat16, one H100 NVL per job (`--gpus=h100:1`; all dev logs checked show H100 NVL),
  transformers/torch versions as recorded in each output file.
- Data: core generator v1.3.0. The partition is fixed: seed 2026, dev fraction .15, 4 test sets.
  Families are unordered name pairs. Test0 has 59 core families (dev 41), zero overlap; BX has 240 test0
  families (dev 169), zero overlap. Core and XL share the same pair assignment. Sampling seed 2026 and
  bootstrap seed 2026 are separate from the partition and are also fixed.
- Validation sets without a test split: UD and WIKI-ROLES. They are reported as validation, not test.
- Code: the freeze commit (patches up to 0069: scorer, `--layers` for OTR-09/OTR-14, submit script, this file).

## 2. Definitions (as computed by the code)
- Bare-answer margin (OTR-09, OTR-14, OTR-07 bare): logit(answer) − logit(distractor) at the answer
  position, first answer token. Pairs whose two names share a first token are excluded.
- Repair = (patched − failing)/(working − failing); damage = (working − patched)/(working − failing).
  Both are per-item ratios, averaged over items. Only items with working − failing > 1.0 enter
  (`--min-gap 1.0`). In OTR-14 `by_from_patient`, the working answer stays the reference.
- OTR-03 dependent variable: the change in the recipient name's margin over the other name after the
  transplant (not the answer-minus-distractor margin). Within-group OLS, centered by base item ID and
  recipient slot; source-slot nuisance included when both roles are asked. Coefficients: order,
  function, role.
- Generation score (C1): `generation_accuracy` from `otr_generate_check.py`, i.e. the proportion of all
  prompts whose reply the extractor credits as the gold name (direct, frame, statement, inferred);
  ambiguous and none count as wrong. Denominator: all prompts.
- Family bootstrap: 2000 resamples of families with replacement, seed 2026; percentile 95% CI. When a
  claim compares two quantities, both are recomputed on the same resampled families (paired). For
  OTR-03 the regression is refitted on every resample.
- "Not evaluable": a measure whose denominator is below the stated minimum or whose fit fails. It
  counts as not passing and is reported as such.

## 3. Claims and pass rules (per model, test0)
| ID | Claim | Measure | Passes if |
|---|---|---|---|
| C1 | The failing question fails selectively | generation score, agent questions, 4 main constructions: working − failing (same item IDs, paired) | CI lower bound > 0 |
| C2 | Roles live on the names | OTR-03 ordinary questions, correct bases, role coefficient at the frozen role layer; role − function at that layer | role CI lower bound > 0, and role − function CI lower bound > 0 |
| C3 | The failing question weakens or reverses the read-out | OTR-03 failing question, agent questions only (`--roles agent`), wrong bases: role coefficient at the frozen role layer, compared with C2 on the same events | C2 role − C3 role: CI lower bound > 0. Minimum 30 wrong bases, otherwise not evaluable. Sign is a secondary prediction: < 0 expected for Qwen3.8, phi-4, OLMo; Gemma near zero (dev agent-only −0.06); Granite weakened (dev +0.19) |
| C4 | A fixed handover depth | OTR-09 residual repair curve, all eligible items (`residual.all.repair`), point estimates on the sampled grid | the first sampled layer with repair >= .5 lies within ±2 layers of the frozen layer (one grid step either way); repair < .5 at every sampled layer up to frozen − 4 |
| C5 | Frozen heads repair; the state is specific | OTR-09 `--rank-from <dev> --skip-scan --n-random 5 --controls`: `transfer.by_k["20"].all_items` (frozen 20 heads, in frozen order) vs `random_same_layers` (5 draws of 20 heads from the same layers, per-item mean) | top − random: CI lower bound > 0; top mean >= .25; `other_failing` and `random_nudge` mean repair <= .10 (signed; fixed in advance, not tuned on dev). Run refuses if any frozen head is dropped |
| C6 | The state at "by" carries the failure | OTR-14 `by_from_failing` damage at the frozen "by" layer, all eligible items. Same intervention and formula as OTR-13 `reverse_damage` for "by", from which the layer is frozen | CI lower bound > 0. (The claim is that the state at "by" carries it, not that only "by" does; OTR-13 showed the end of the question also carries it.) |
| C7 | Knows but answers wrong (Gemma, Granite, Qwen3.8 only) | OTR-07, failing condition only. Denominator: items with `correct == false` (bare margin picks the patient). Numerator: those with `ld_model_by > 0` after the neutral prefix "Someone was V-ed by" (0 counts as not agent) | share > .5 and CI lower bound > .5. Minimum 20 items, otherwise not evaluable. No claim for OLMo, phi-4 |
| C8 | "by" from the patient question damages the working answer | OTR-14 `by_from_patient`, pooled items, at the frozen "by" layer | CI lower bound > 0 (all five models: dev CI lower bound > 0 at that layer, see section 7) |

Reporting: 8 claims x 5 models (C7 for three). Each claim is reported as the number of models passing, with every
estimate and CI shown. No multiple-comparison correction is applied to these replication criteria; the
paper says so. A failed or not-evaluable criterion is reported as such.

Exploratory, not tested on test0: layer-pipeline ordering (OTR-14 read windows), prompt robustness
(OTR-15), cross-set transfer, steering vector, psych and ditransitive blocks, all UD and WIKI results.

## 4. Frozen per-model settings (dev only; never re-chosen)
Layers are physical layer indices. Dev peak rule: the layer with the highest dev value (OTR-03: role
coefficient of the full dev fit; OTR-13: mean "by" reverse damage); ties go to the earlier layer. C4's tolerance of
±2 layers is one grid step; on the dev-like WIKI validation set (24 Nov) the handover layer reappeared
exactly in Gemma, Granite and phi-4, and 2–4 layers earlier in OLMo.

| Model | layers | handover (first dev crossing of .5) | role layer (OTR-03 dev peak) | "by" layer (OTR-13 dev peak of "by" reverse damage) | dev OTR-09 file (`--rank-from`) | HF revision |
|---|---|---|---|---|---|---|
| Gemma 4 31B | 60 | 40 | 28 | 28 | `otr09_google__gemma-4-31B-it_dev_20261118T164326.json` | `842da3794eaa` |
| Qwen3.8-27B | 64 | 54 | 29 | 30 | `otr09_Qwen__Qwen3.8-27B_dev_20261118T162630.json` | `37561816e66b` |
| phi-4 | 40 | 22 | 15 | 8 | `otr09_microsoft__phi-4_dev_20261119T093535.json` | `2db69c1c3e91` |
| OLMo-3.1-32B | 64 | 26 | 18 | 8 | `otr09_allenai__Olmo-3.1-32B-Instruct_dev_20261119T095922.json` | `ac0587e4a774` |
| Granite-4.2-30B | 64 | 52 | 37 | 28 | `otr09_ibm-granite__granite-4.2-30b_dev_20261119T140015.json` | `9e668ce1c538` |

Codex checked the five handover layers, the five top-20 lists and phi-4's role layer against the
archived dev files: all match. Full revision hashes: Gemma 842da3794eaa0b77d5f08bae87a17459d91ff475,
Qwen3.8 37561816e66b5b9153b1b64c10968c18c1e6f4a0, phi-4 2db69c1c3e91a05d2c64a3185acfbaf36f744e25, OLMo
ac0587e4a7744a551c059d8cd17ba220bc940dae, Granite 9e668ce1c538387ef24d3644e9b0606647762636. The sha256 of every dev
file the settings were read from is in `docs/order_to_role/freeze_inputs.sha256`.

Frozen top-20 heads (dev OTR-09 `head_scan.top20`, in this order):
- Gemma: 49.16,40.12,39.23,56.4,58.26,46.29,40.1,41.14,49.31,57.13,47.17,58.19,39.6,41.1,40.7,49.11,46.30,39.2,41.8,53.18
- Qwen3.8: 59.3,63.22,51.23,59.21,59.9,63.21,51.20,47.4,59.7,47.17,51.4,51.16,63.0,55.2,59.6,51.19,47.3,63.14,63.3,55.15
- phi-4: 26.26,25.20,30.17,32.23,25.13,24.16,25.18,23.28,23.33,23.1,35.19,30.19,30.18,32.20,24.23,22.28,35.4,26.28,24.24,38.22
- OLMo-3.1: 49.19,48.33,54.25,49.15,47.18,35.7,63.2,35.37,61.33,41.6,56.23,57.31,54.28,51.4,51.0,61.31,50.27,56.20,51.2,60.29
- Granite: 51.27,58.6,55.23,52.25,61.27,55.21,49.26,51.6,50.23,63.15,43.28,47.26,62.30,52.9,49.24,51.10,51.14,51.12,59.27,61.24

## 5. Test-run commands (one batch, run once)
All test jobs are submitted by `scripts/slurm/otr_test0_submit.sh --submit` (run it without `--submit`
first: it prints every command). On Pelle, in `~/private/llm_relation`, at the freeze commit, with the
five dev OTR-09 files of section 4 present under `results/`. Every job: `sbatch -A uppmax2026-1-225
--gpus=h100:1`, `--chat-template --split test0`, `--no-thinking` for Qwen3.8 and Granite. To save GPU time
the runs compute only the layers the claims read (the per-layer curves for figures come from dev):
- C1: `otrgen_pelle.sbatch --n 512 --instruction "Reply with one name only." --question-role agent`, once with
  `--question-template "Who was someone {past} by?"` and once with `"Who was the {ing} done by?"`.
- C2: `otr03_pelle.sbatch --n-events 64 --layers <role layer>`.
- C3: `otr03_pelle.sbatch --n-events 64 --layers <role layer> --roles agent --agent-question "Who was someone {past} by?" --include-incorrect`.
- C4/C5: `otr09_pelle.sbatch --instruction "Reply with one name only." --rank-from <dev OTR-09 file> --skip-scan --n-random 5 --controls --layers 0,2,...,<handover+2>`.
- C6/C8: `otr14_pelle.sbatch --rank-from <dev OTR-09 file> --layers <"by" layer>`.
- C7 (Gemma, Granite, Qwen3.8): `otr07_pelle.sbatch --instruction "Reply with one name only." --by-prefix "Someone was {verb} by" --condition agent "Who was someone {past} by?" --condition agent "Who was the {ing} done by?"`.
- Replication (BX transitive, after the core batch; reported separately from the core pass counts):
  C2 and C3 with `--stimuli bx-transitive --n-events 48 --layers <role layer>`, C5 with `--stimuli bx-transitive`
  (frozen core heads, no controls). Scored with `--stimuli bx-transitive`.
- Spanish/Galician (XL) are not run on test0; they stay dev-only validation.
- Sample: 64 events cover 42 test0 families (C2/C3).
- Queue order: right after submitting, put every pending dev job on hold (`scontrol hold`) or move the
  test jobs to the top (`scontrol top`), so test0 runs first.
- The script appends each job ID to `docs/order_to_role/test0_attempts.md`.

## 6. Scoring, failures, log
- Scorer: `scripts/otr_freeze_score.py` v1.0.0 (`--split test0`; `--stimuli bx-transitive` for the
  replication) reads the outputs and prints the pass/fail table for C1–C8 by the rules above (family
  from the item ID, paired family bootstrap with 2000 resamples and seed 2026, minimums, missing layers or
  fields = not evaluable). Tests: `tests/test_freeze_score.py`.
- Dev dry run (checks the code path; C4, C6 and C8 read the same dev data the layers came from, so this
  is not evidence). C5 is not evaluable on dev until the dev controls run returns; its threshold is
  fixed in advance:

| Claim | gemma | Qwen3.8 | phi | Olmo | granite |
|---|---|---|---|---|---|
| C1 | PASS (0.31) | PASS (0.45) | PASS (0.79) | PASS (0.31) | PASS (0.19) |
| C2 | PASS (1.08) | PASS (0.58) | PASS (0.59) | PASS (0.53) | PASS (0.75) |
| C3 | PASS (1.14) | PASS (0.73) | PASS (1.20) | PASS (0.73) | PASS (0.56) |
| C4 | PASS (40) | PASS (54) | PASS (22) | PASS (26) | PASS (52) |
| C5 | not evaluable (transfer k=20 missing) | not evaluable (transfer k=20 missing) | not evaluable (transfer k=20 missing) | not evaluable (transfer k=20 missing) | not evaluable (transfer k=20 missing) |
| C6 | PASS (0.81) | PASS (0.58) | PASS (0.34) | PASS (1.12) | PASS (1.53) |
| C7 | PASS (1.00) | PASS (0.83) | no claim | no claim | PASS (0.85) |
| C8 | PASS (0.13) | PASS (0.67) | PASS (0.25) | PASS (0.99) | PASS (1.08) |

- Technical failure (crash, timeout, out of memory, partial output): rerun the identical command,
  at most twice more. If it still fails, the claim is reported as not evaluable for that model. Every
  submission, job ID and outcome goes in `docs/order_to_role/test0_attempts.md`.
- After test0 is opened nothing in sections 2–6 changes. Any further analysis is labelled exploratory.

## 7. Decisions recorded before freezing
- Spanish/Galician judgements were not settled before freezing, so XL is not part of test0.
- Done 24 Nov: role layers from the dev OTR-03 peaks (argmax of the dev role profile, full layer grid):
  Gemma 28, Qwen3.8 29, phi-4 15, OLMo 18, Granite 37. Near-ties: Qwen3.8 30 and 42, Granite 38 and 52
  (Granite's profile is flat from about layer 37 to 52); the argmax rule decides.
- Done 25 Nov: "by" layers from OTR-13 dev (reverse damage of "by", argmax on the even-layer grid):
  Gemma 28 (.78), Qwen3.8 30 (.59), phi-4 8 (.35), OLMo 8 (1.12), Granite 28 (1.55). OTR-14 samples
  even layers by default, so all five are on its grid. C5's .10 threshold fixed in advance.
- Done 26 Nov: dev OTR-14 arrived. `by_from_failing` peaks at exactly the OTR-13 layers in all five
  models (same intervention; consistency check, not independent evidence): Gemma .81 [.33, 1.62],
  Qwen3.8 .58 [.51, .67], phi-4 .34 [.25, .43], OLMo 1.12 [.88, 1.43], Granite 1.53 [1.20, 1.91].
  `by_from_patient` at the frozen "by" layer: Gemma .13 [.07, .23], Qwen3.8 .67 [.55, .80], phi-4 .25
  [.16, .35], OLMo .99 [.83, 1.17], Granite 1.08 [.82, 1.40]; CI lower bound > 0 in all five, so C8 is
  a tested claim for all five (decided on dev, before freezing).
- Done 25 Nov: dev C3 recheck with agent questions only (wrong bases, role coefficient at the role
  layer): phi-4 −0.60 (94 bases), OLMo −0.20 (47), Qwen3.8 −0.15 (59), Gemma −0.06 (85), Granite
  +0.19 (32). The old fits mixed in wrong patient answers. phi-4's correct agent items are also
  negative (−0.52), so C3 is stated as weaker read-out, not as the cause of each error.
- Done 26 Nov: scorer written and dry-run on dev (section 6); revision hashes and dev-file sha256 recorded.
- Joel: final read of sections 3–5 before committing.
