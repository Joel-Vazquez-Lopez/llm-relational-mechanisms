# FREEZE (DRAFT v2, 24 Nov 2026) — order-to-role, test set 0

Status: DRAFT v2. Self-reviewed by Joel and Claude, and reviewed adversarially by Codex against the
code (24 Nov; 11 points, all addressed below or listed as open). There is no supervisor sign-off; the
public GitHub commit timestamp is the record. This file becomes `docs/order_to_role/FREEZE.md`; the
commit that adds it (together with the scorer, section 6) must come before any `--split test0` run.
Items marked TBD are filled from dev runs only. Nothing here is chosen on test data.

## 0. Prior access to held-out data (as of 24 Nov)
- No result file with a test split or status `TEST_SET_OPENED` exists in the results repository
  (checked 24 Nov: 0 files). The older OTR-01 runner has a `--split test --test-set 0` option; it was
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
- Model revisions: the Hugging Face snapshot hash of each model, as cached on Pelle, goes in the table
  below before freezing (TBD: `ls /proj/uppmax2026-1-225/hf_cache/hub/models--*/snapshots`).
  Runtime: bfloat16, one H100, transformers/torch versions as recorded in each output file.
- Data: core generator v1.3.0. The partition is fixed: seed 2026, dev fraction .15, 4 test sets.
  Families are unordered name pairs. Test0 has 59 core families (dev 41), zero overlap; BX has 240 test0
  families (dev 169), zero overlap. Core and XL share the same pair assignment. Sampling seed 2026 and
  bootstrap seed 2026 are separate from the partition and are also fixed.
- Validation sets without a test split: UD and WIKI-ROLES. They are reported as validation, not test.
- Code: the freeze commit, i.e. patches up to 0066 plus the scorer plus this file.

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
| C3 | The failing question weakens or reverses the read-out | OTR-03 failing question, agent questions only (`--roles agent`), wrong bases: role coefficient at the frozen role layer, compared with C2 on the same events | C2 role − C3 role: CI lower bound > 0. Minimum 30 wrong bases, otherwise not evaluable. Sign is a secondary prediction: < 0 expected for Qwen3.8, phi-4, OLMo; Gemma may stay > 0 at its role layer (its dev reversal was in middle layers); Granite weakened |
| C4 | A fixed handover depth | OTR-09 residual repair curve, all eligible items (`residual.all.repair`), point estimates on the sampled grid | the first sampled layer with repair >= .5 lies within ±2 layers of the frozen layer (one grid step either way); repair < .5 at every sampled layer up to frozen − 4 |
| C5 | Frozen heads repair; the state is specific | OTR-09 `--rank-from <dev> --skip-scan --n-random 5 --controls`: `transfer.by_k["20"].all_items` (frozen 20 heads, in frozen order) vs `random_same_layers` (5 draws of 20 heads from the same layers, per-item mean) | top − random: CI lower bound > 0; top mean >= .25; `other_failing` and `random_nudge` mean repair <= .10 (signed). Run refuses if any frozen head is dropped |
| C6 | The state at "by" carries the failure | OTR-14 `by_from_failing` damage at the frozen "by" layer, all eligible items | CI lower bound > 0. (The claim is that the state at "by" carries it, not that only "by" does; OTR-13 showed the end of the question also carries it.) |
| C7 | Knows but answers wrong (Gemma, Granite, Qwen3.8 only) | OTR-07, failing condition only. Denominator: items with `correct == false` (bare margin picks the patient). Numerator: those with `ld_model_by > 0` after the neutral prefix "Someone was V-ed by" (0 counts as not agent) | share > .5 and CI lower bound > .5. Minimum 20 items, otherwise not evaluable. No claim for OLMo, phi-4 |
| C8 | "by" from the patient question damages the working answer | OTR-14 `by_from_patient`, pooled items, at the frozen "by" layer | Included only for models whose dev CI lower bound is > 0 (decided before freezing, listed in section 4). Pass: CI lower bound > 0 |

Reporting: 8 claims x 5 models. Each claim is reported as the number of models passing, with every
estimate and CI shown. No multiple-comparison correction is applied to these replication criteria; the
paper says so. A failed or not-evaluable criterion is reported as such.

Exploratory, not tested on test0: layer-pipeline ordering (OTR-14 read windows), prompt robustness
(OTR-15), cross-set transfer, steering vector, psych and ditransitive blocks, all UD and WIKI results.

## 4. Frozen per-model settings (dev only; never re-chosen)
Layers are physical layer indices. Dev peak rule: the layer with the highest dev value (OTR-03: role
coefficient of the full dev fit; OTR-14: mean damage); ties go to the earlier layer. C4's tolerance of
±2 layers is one grid step; on the dev-like WIKI validation set (24 Nov) the handover layer reappeared
exactly in Gemma, Granite and phi-4, and 2–4 layers earlier in OLMo.

| Model | layers | handover (first dev crossing of .5) | role layer (OTR-03 dev peak) | "by" layer (OTR-14 dev peak) | C8 included | dev OTR-09 file (`--rank-from`) | HF revision |
|---|---|---|---|---|---|---|---|
| Gemma 4 31B | 60 | 40 | 28 | TBD | TBD | `otr09_google__gemma-4-31B-it_dev_20261118T164326.json` | TBD |
| Qwen3.8-27B | 64 | 54 | 29 | TBD | TBD | `otr09_Qwen__Qwen3.8-27B_dev_20261118T162630.json` | TBD |
| phi-4 | 40 | 22 | 15 | TBD | TBD | `otr09_microsoft__phi-4_dev_20261119T093535.json` | TBD |
| OLMo-3.1-32B | 64 | 26 | 18 | TBD | TBD | `otr09_allenai__Olmo-3.1-32B-Instruct_dev_20261119T095922.json` | TBD |
| Granite-4.2-30B | 64 | 52 | 37 | TBD | TBD | `otr09_ibm-granite__granite-4.2-30b_dev_20261119T140015.json` | TBD |

Codex checked the five handover layers, the five top-20 lists and phi-4's role layer against the
archived dev files: all match. The sha256 of every dev file used goes into `docs/order_to_role/
freeze_inputs.sha256` with the freeze commit.

Frozen top-20 heads (dev OTR-09 `head_scan.top20`, in this order):
- Gemma: 49.16,40.12,39.23,56.4,58.26,46.29,40.1,41.14,49.31,57.13,47.17,58.19,39.6,41.1,40.7,49.11,46.30,39.2,41.8,53.18
- Qwen3.8: 59.3,63.22,51.23,59.21,59.9,63.21,51.20,47.4,59.7,47.17,51.4,51.16,63.0,55.2,59.6,51.19,47.3,63.14,63.3,55.15
- phi-4: 26.26,25.20,30.17,32.23,25.13,24.16,25.18,23.28,23.33,23.1,35.19,30.19,30.18,32.20,24.23,22.28,35.4,26.28,24.24,38.22
- OLMo-3.1: 49.19,48.33,54.25,49.15,47.18,35.7,63.2,35.37,61.33,41.6,56.23,57.31,54.28,51.4,51.0,61.31,50.27,56.20,51.2,60.29
- Granite: 51.27,58.6,55.23,52.25,61.27,55.21,49.26,51.6,50.23,63.15,43.28,47.26,62.30,52.9,49.24,51.10,51.14,51.12,59.27,61.24

## 5. Test-run commands (one batch, run once)
On Pelle, in `~/private/llm_relation`, at the freeze commit. `$M` = model ID; `$X` = `--no-thinking` for
Qwen3.8 and Granite, empty otherwise; `$RF` = the model's dev OTR-09 file from section 4 (under
`results/`); `$GRID` = even layers 0,2,...,L−2 plus the model's role layer. Every command:
`sbatch -A uppmax2026-1-225 scripts/slurm/<script> $M $X --chat-template --split test0 ...`
(GPU and time requests are in each sbatch file).
- C1: `otrgen_pelle.sbatch $M $X --chat-template --split test0 --n 512 --instruction "Reply with one name only." --question-template "Who was someone {past} by?" --question-role agent`, and the same with `"Who was the {ing} done by?"`
- C2: `otr03_pelle.sbatch $M $X --chat-template --split test0 --n-events 64 --layers $GRID`
- C3: `otr03_pelle.sbatch $M $X --chat-template --split test0 --n-events 64 --layers $GRID --roles agent --agent-question "Who was someone {past} by?" --include-incorrect`
- C4/C5: `otr09_pelle.sbatch $M $X --chat-template --split test0 --instruction "Reply with one name only." --rank-from $RF --skip-scan --n-random 5 --controls`
- C6/C8: `otr14_pelle.sbatch $M $X --chat-template --split test0 --rank-from $RF`
- C7 (Gemma, Granite, Qwen3.8): `otr07_pelle.sbatch $M $X --chat-template --split test0 --instruction "Reply with one name only." --by-prefix "Someone was {verb} by" --condition agent "Who was someone {past} by?" --condition agent "Who was the {ing} done by?"`
- Sample: 64 events cover 42 test0 families (C2/C3). The other runs use their default sizes.
- Replication sets: BX transitive test0, C2, C3 and C5 (same commands with `--stimuli bx-transitive`
  and the BX failing question; C5 uses the core frozen heads and `$RF`, so it tests transfer of the
  frozen heads to new names, verbs and constructions on held-out families). Reported as replication,
  separately from the core pass counts. Run after the core batch. XL: only constructions cleared by Joel; if only active and passive are
  cleared, XL is not run on test0 (the XL OTR-03 path uses four constructions), and XL stays dev-only.

## 6. Scoring, failures, log
- Scorer: `scripts/otr_freeze_score.py` reads the test0 outputs and prints the pass/fail table for
  C1–C8 by the rules above (family joins, paired bootstrap, minimums, ties, missing layers). It is
  committed with the freeze and dry-run on the dev outputs first; the dev table goes in the freeze
  commit message.
- Technical failure (crash, timeout, out of memory, partial output): rerun the identical command,
  at most twice more. If it still fails, the claim is reported as not evaluable for that model. Every
  submission, job ID and outcome goes in `docs/order_to_role/test0_attempts.md`.
- After test0 is opened nothing in sections 2–6 changes. Any further analysis is labelled exploratory.

## 7. Open before freezing
- Joel: es/gl judgements (by-cleft causal reading; Galician "A María" OVS) -> which XL constructions enter.
- Done 24 Nov: role layers from the dev OTR-03 peaks (argmax of the dev role profile, full layer grid):
  Gemma 28, Qwen3.8 29, phi-4 15, OLMo 18, Granite 37. Near-ties: Qwen3.8 30 and 42, Granite 38 and 52
  (Granite's profile is flat from about layer 37 to 52); the argmax rule decides.
- "By" layers and the C8 inclusion list (dev OTR-14, arriving).
- Check C5's control threshold (.10) against the dev controls run (arriving); change only before freezing.
- Write and dry-run the scorer on dev; recheck dev C3 with agent questions only (from the saved records).
- HF revision hashes; sha256 of the dev input files.
- Joel + Claude: final read of sections 3–5.
