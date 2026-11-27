[← Order-to-Role](README.md) · [Results](RESULTS.md) · [Methods](METHODS.md)

# Evidence ledger

Every claim of the study is listed here with its status: confirmed on held-out data, exploratory, failed, inconclusive, dropped or ongoing. Negative results stay on the record.

**Snapshot:** code `9c43792`, results `25c6914` (27 November 2026). The confirmatory tables are the frozen scorer's output on the archived test0 files. They were produced for this page by [`analysis/build_results.py`](analysis/build_results.py), which calls `scripts/otr_freeze_score.py` v1.1.0 unchanged. No model was run and no further test set was opened.

---

## 1. Confirmatory results (test set 0)

**English:** 37 of 38 checks pass (1 fail, 2 cells with no claim). **BX replication:** 15 of 15. **Spanish:** 29 of 30. **Galician:** 30 of 30.

Each cell gives the verdict, the frozen primary estimate and its 95% family-bootstrap interval (2,000 resamples, seed 2026). Pass rules: [FREEZE.md §3](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE.md) and [FREEZE_XL.md §3](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE_XL.md). C5's verdict also depends on the controls, and C2 also requires role > function in English. Machine-readable: [`confirmatory_test0.csv`](results/tables/confirmatory_test0.csv); full scorer output with input checksums: [`results/confirmatory/`](results/confirmatory/).

#### English core

| Claim | Gemma | Qwen3.8 | phi-4 | OLMo | Granite |
|:--|--|--|--|--|--|
| C1 failing question fails selectively (working − failing) | ✅ 0.28 [0.23, 0.33] | ✅ 0.47 [0.44, 0.51] | ✅ 0.77 [0.73, 0.81] | ✅ 0.36 [0.32, 0.39] | ✅ 0.16 [0.13, 0.20] |
| C2 role on the names (role coefficient) | ✅ 1.42 [1.27, 1.57] | ✅ 0.62 [0.58, 0.67] | ✅ 0.45 [0.40, 0.50] | ✅ 0.49 [0.45, 0.54] | ✅ 0.60 [0.54, 0.66] |
| C3 read-out weaker on wrong items (C2 − C3) | ✅ 1.53 [1.32, 1.72] | ✅ 0.82 [0.75, 0.90] | ✅ 1.15 [1.06, 1.23] | ✅ 0.68 [0.59, 0.77] | ✅ 0.45 [0.35, 0.53] |
| C4 handover at the frozen layer (first crossing) | ✅ L40 | ✅ L54 | ✅ L22 | ✅ L26 | ✅ L52 |
| C5 frozen heads beat random (top − random) | ✅ 0.75 [0.74, 0.77] | ✅ 0.36 [0.34, 0.39] | ❌ 0.72 [0.70, 0.74] | ✅ 0.39 [0.36, 0.41] | ✅ 0.54 [0.50, 0.58] |
| C6 “by” carries the failure (damage) | ✅ 0.76 [0.25, 1.57] | ✅ 0.57 [0.45, 0.67] | ✅ 0.41 [0.32, 0.49] | ✅ 1.27 [1.02, 1.55] | ✅ 1.33 [1.00, 1.75] |
| C7 knows but answers wrong (agent share) | ✅ 1.00 [1.00, 1.00] | ✅ 0.82 [0.80, 0.84] | no claim | no claim | ✅ 0.84 [0.78, 0.90] |
| C8 patient-question “by” damages (damage) | ✅ 0.16 [0.07, 0.28] | ✅ 0.61 [0.51, 0.71] | ✅ 0.33 [0.22, 0.42] | ✅ 1.06 [0.81, 1.32] | ✅ 0.98 [0.68, 1.37] |

#### English BX replication

| Claim | Gemma | Qwen3.8 | phi-4 | OLMo | Granite |
|:--|--|--|--|--|--|
| C2 role on the names (role coefficient) | ✅ 1.19 [1.05, 1.32] | ✅ 0.67 [0.61, 0.72] | ✅ 0.53 [0.46, 0.59] | ✅ 0.54 [0.49, 0.60] | ✅ 0.73 [0.65, 0.81] |
| C3 read-out weaker on wrong items (C2 − C3) | ✅ 1.05 [0.87, 1.23] | ✅ 0.84 [0.75, 0.95] | ✅ 1.31 [1.17, 1.44] | ✅ 0.85 [0.70, 0.98] | ✅ 0.51 [0.42, 0.59] |
| C5 frozen heads beat random (top − random) | ✅ 0.71 [0.70, 0.73] | ✅ 0.34 [0.31, 0.36] | ✅ 0.72 [0.70, 0.74] | ✅ 0.33 [0.29, 0.36] | ✅ 0.49 [0.45, 0.53] |

#### Spanish (XL addendum, primary)

| Claim | Gemma | Qwen3.8 | phi-4 | OLMo | Granite |
|:--|--|--|--|--|--|
| XL-C1 failing question fails selectively (working − failing) | ✅ 0.30 [0.25, 0.35] | ✅ 0.32 [0.26, 0.38] | ✅ 0.80 [0.75, 0.85] | ✅ 0.68 [0.62, 0.74] | ✅ 0.50 [0.43, 0.57] |
| XL-C2 role on the names (role coefficient) | ✅ 0.67 [0.57, 0.77] | ✅ 0.74 [0.67, 0.80] | ✅ 0.35 [0.31, 0.39] | ✅ 0.41 [0.35, 0.48] | ✅ 0.50 [0.45, 0.55] |
| XL-C3 read-out weaker on wrong items (C2 − C3) | ❌ 0.11 [-0.01, 0.23] | ✅ 0.69 [0.61, 0.77] | ✅ 0.67 [0.60, 0.74] | ✅ 0.65 [0.55, 0.75] | ✅ 0.56 [0.48, 0.63] |
| XL-C4 handover at the frozen layer (first crossing) | ✅ L48 | ✅ L50 | ✅ L22 | ✅ L24 | ✅ L52 |
| XL-C5 frozen heads beat random (top − random) | ✅ 0.71 [0.69, 0.73] | ✅ 0.31 [0.30, 0.33] | ✅ 0.62 [0.61, 0.64] | ✅ 0.42 [0.41, 0.44] | ✅ 0.55 [0.54, 0.57] |
| XL-C5b repair is item-specific (own − other working) | ✅ 4.08 [3.12, 5.05] | ✅ 0.62 [0.55, 0.71] | ✅ 0.47 [0.44, 0.50] | ✅ 0.39 [0.34, 0.44] | ✅ 0.49 [0.43, 0.56] |

#### Galician (XL addendum, secondary)

| Claim | Gemma | Qwen3.8 | phi-4 | OLMo | Granite |
|:--|--|--|--|--|--|
| XL-C1 failing question fails selectively (working − failing) | ✅ 0.50 [0.44, 0.55] | ✅ 0.39 [0.32, 0.46] | ✅ 0.49 [0.41, 0.57] | ✅ 0.53 [0.46, 0.60] | ✅ 0.38 [0.31, 0.46] |
| XL-C2 role on the names (role coefficient) | ✅ 0.90 [0.80, 0.99] | ✅ 0.75 [0.69, 0.82] | ✅ 0.35 [0.32, 0.39] | ✅ 0.19 [0.15, 0.22] | ✅ 0.45 [0.40, 0.49] |
| XL-C3 read-out weaker on wrong items (C2 − C3) | ✅ 1.47 [1.32, 1.63] | ✅ 0.73 [0.66, 0.81] | ✅ 0.52 [0.47, 0.58] | ✅ 0.27 [0.22, 0.31] | ✅ 0.30 [0.24, 0.34] |
| XL-C4 handover at the frozen layer (first crossing) | ✅ L48 | ✅ L50 | ✅ L22 | ✅ L26 | ✅ L52 |
| XL-C5 frozen heads beat random (top − random) | ✅ 0.74 [0.72, 0.75] | ✅ 0.29 [0.28, 0.31] | ✅ 0.60 [0.57, 0.63] | ✅ 0.37 [0.35, 0.39] | ✅ 0.56 [0.53, 0.58] |
| XL-C5b repair is item-specific (own − other working) | ✅ 2.73 [2.00, 3.53] | ✅ 0.43 [0.37, 0.50] | ✅ 0.61 [0.49, 0.74] | ✅ 0.34 [0.29, 0.38] | ✅ 0.69 [0.55, 0.83] |

### Notes on the confirmatory results

<a id="phi-4-c5"></a>**phi-4 fails C5 on a control, not on repair.** Its frozen heads repair .72 against .01 for random heads. C5 also required the `other_failing` control (head outputs from another item's failing run) to stay at or below .10. In phi-4 it repaired .26. Replacing the heads' outputs with any other item's output removes their own wrong-way push, so this control does not test item-specific content. The Spanish/Galician addendum therefore scores a different content control (C5b: own working-run output vs another item's), which phi-4 passes in both languages. FREEZE.md stays as it was, and the English failure stands.

**Gemma fails XL-C3 in Spanish.** Its read-out under the failing question is weaker than under ordinary questions by +0.11, but the interval includes zero. The addendum had written this down in advance from development data (C2 − C3 = .08 on development).

**Secondary sign predictions held.** FREEZE.md predicted the C3 role coefficient on wrong items would be negative for Qwen3.8, phi-4 and OLMo, near zero for Gemma and weakened but positive for Granite. Test0 gives −0.20, −0.70, −0.19, −0.11 (interval includes zero) and +0.15.

**Scale caveat for XL-C5b in Gemma.** The values 4.08 (Spanish) and 2.73 (Galician) are large because another item's working-run output pushes Gemma's answer strongly the wrong way (`other_failing` −3.36 and −1.93). The verdicts are valid, but the magnitudes are not comparable across models.

**Provenance.** The canonical attempt logs are in `results/` on `main` (`results/test0_attempts_pelle.md`, with `results/test0_attempts.md`), imported byte for byte from the results branch. They list 48 English submissions at the freeze commit `f6ea0e1` (26 Nov, 10:10) and 40 Spanish/Galician submissions at `c40a1f7` (26 Nov, 19:11). `c40a1f7` only adds the native-speaker check that the addendum requires before submitting. No reruns are recorded. `docs/order_to_role/test0_attempts.md` is the empty template that the submit script was meant to fill.

## 2. Exploratory findings (development only)

These shaped the frozen claims or extend them, but were not tested on held-out data.

| Finding | Evidence | Where |
|:--|:--|:--|
| Role is the largest transplant coefficient in all five models; order/focus carries a smaller signal in Gemma, phi-4 and Granite | OTR-03 core and BX | [Results §2](RESULTS.md#2-the-names-carry-their-thematic-role) |
| Psych verbs: the code follows the experiencer, not the subject | OTR-03 BX psych (results log) | [Results §2](RESULTS.md#2-the-names-carry-their-thematic-role) |
| Same layer order in every model: role and "by" peaks before the handover | OTR-03, OTR-13, OTR-09, OTR-14 | [Layers](LAYERS.md) |
| Whole-question swap stops repairing exactly where answer-position patching starts | OTR-14 | [Layers](LAYERS.md) |
| Heads frozen on templates repair BX, UD and Wikipedia sentences; 13–19 of 20 overlap with freshly ranked heads | OTR-09 transfer | [Generality](TRANSFER.md) |
| The failing question fails on natural text too; real and neutral names agree | OTR-09 on UD and WIKI-ROLES | [Generality](TRANSFER.md) |
| Ditransitive and psych blocks also fail under the stranded "by" question | Generation, BX | [Generality](TRANSFER.md) |
| Knockout of the selected heads and an averaged role vector | OTR-12 | [Answers and interventions](INTERVENTIONS.md) |
| Prompt-format robustness of the generation effect | OTR-15 | [Answers and interventions](INTERVENTIONS.md) |
| Llama 3.3 70B and Qwen2.5 72B show a handover and repairing heads | OTR-03, OTR-09 (dev, 27 Nov) | [Results §8](RESULTS.md#8-larger-models-exploratory) |

## 3. Failed predictions and negative results

| What was expected or claimed | What happened | Consequence |
|:--|:--|:--|
| Models would pass a strict behavioural gate on all constructions | Qwen2.5 1.5B/7B (base and instruct) failed (.55–.65 overall); gpt-oss-20b missed the inclusion gate | Model set changed; gpt-oss reported descriptively only |
| Forced-choice scoring reveals a human-like subject–object asymmetry | It was an artefact of chat models restating the sentence | Claim dropped; generation scoring adopted ([Results §1](RESULTS.md#1-behaviour-one-question-wording-fails-and-scoring-matters)) |
| A staged order → function → role conversion across layers (the original "order-to-role" hypothesis) | Not seen. In Qwen2.5-7B base, transplants were dominated by name identity (role ≈ 0.02 logits); capable models encode role directly in the middle layers | Hypothesis dropped; the study's name is historical |
| Information about the answer at the verb, "was", "by" or the period (OTR-02, Qwen2.5) | None at any layer; the answer was read late from the name's own position | Led to the transplant and answer-position designs |
| The verb would carry the question's instruction (OTR-13) | The state at "by" carries it; the verb carries little (phi-4 on BX is an exception) | Claim C6 written about "by" |
| The first neutral-prefix test (prefix naming the patient) shows the model knows the agent | The patient's name in the prefix could do the work | Replaced by "Someone was V-ed by"; C7 restricted to Gemma, Qwen3.8 and Granite |
| phi-4 C5 control ≤ .10 | .26 [test0] | Reported as a fail; control replaced in the addendum |
| Gemma Spanish XL-C3 | Interval includes zero [test0] | Reported as a fail (predicted from development) |

## 4. Inconclusive or open

- **Order versus focus.** In the four main constructions the first-mentioned name is also the focused one, so the order coefficient cannot be split further.
- **The end of the question carries part of the failure.** "by" is the main entry point, not the only one, and with a second working question Gemma's difference moves to "?" (development).
- **Which questions fail, in general.** The study characterises one family of failing questions. A battery of natural wordings ("Who was Mary chased by?", "By whom was someone chased?") was running at the 24 November snapshot.
- **BX `long_object_cleft` anomaly.** Several models score low on ordinary questions for this construction but high under the failing question. Raw outputs still need to be inspected before the construction is used.
- **Qwen3.8's random-head baseline** is the highest among the models (.11 on held-out development items), and negative on several transfer sets.
- **A complete native pathway.** Interventions show sufficiency under patching. Knockout (OTR-12, development) shows partial necessity: 51–82% of the agent margin is lost in four models (16% in OLMo), but most answers survive. Exclusivity is not tested.

## 5. Ongoing

- **70B-class extension.** Llama 3.3 70B (OTR-03, OTR-09) and Qwen2.5 72B (OTR-09) development runs exist; Qwen2.5 72B OTR-03 is not yet run. A separate freeze addendum is required before any test0 run ([plan](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/BIG_MODELS.md)).
- **Swedish.** Excluded until a native speaker completes the [check sheet](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/swedish_check_sheet.md).
- **A normative Galician variant** (without the prepositional accusative), possibly as an exploratory run.
- **Manuscript figures and intervals** for the development analyses.
- **Test sets 1–3 remain unopened.**

## 6. Corrections and disclosures

- **Donor leak in development OTR-03.** Development donor name pairs were drawn from all pairs, including test families; donors are counterbalanced and no donor-level result was inspected. Test runs draw donors from their own split only and abort on a leak (FREEZE.md §0).
- **The Spanish/Galician addendum was frozen after English test0 had been seen**, but before any Spanish or Galician test run. Its values come from development es/gl runs, listed with checksums.
- **Some 16 November results were committed to the stimuli branch by mistake** (commits `91e66ca`, `e1bff8c`, `e725b8f`); their final versions are identical to copies on `main`, and the earlier versions (including the reverted 16 November run) remain in `main`'s history through the PR #19 merge.
- **Repair is a conditional estimand.** It is computed on items with a working–failing gap above 1 logit, and is not an accuracy.
- **No multiplicity correction** is applied to the replication criteria, as stated in the freeze.
- **This ledger replaces an earlier statement** in the shared research guide that test0 scoring had not been audited.
