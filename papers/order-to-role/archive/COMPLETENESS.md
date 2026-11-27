[← Archive](README.md) · [Paper II](../README.md) · [Experiment map](../EXPERIMENTS.md)

# Completeness of the Order-to-Role record

This page checks, after the results-consolidation PR, that `main` together with the verified archive holds the complete record of every Order-to-Role study. Nothing was run to produce it: it lists what exists. Code paths are on `main`. "Results" counts the result JSONs in `results/`; each also has a plot and, for runs from 22 November on, a Slurm log.

**Status labels.** *Confirmatory*: frozen and tested once on test set 0. *Completed (development)*: run on the development split and summarised. *Exploratory*: run, not frozen, partly or not summarised. *Superseded*: replaced by a later study. *Failed*: a frozen prediction that did not hold. *Not evaluable*: the rule could not be applied. *Proposed*: planned, not run. Nothing proposed is presented as completed.

## Studies

| Study | Code (on `main`) | Results on `main` | Models | Status |
|:--|:--|:--|:--|:--|
| Generation check | `scripts/otr_generate_check.py`, `otr_rescore_generation.py` | 245 dev (core, BX, es, gl, stranded-P, sv) + 20 test0 | 12 models, incl. the five core | **Confirmatory** C1 (English 5/5), XL-C1 (es 5/5, gl 5/5) |
| OTR-01 behaviour gate | `scripts/otr01_behavior_gate.py` | 21 dev | 16 models | Completed (development): inclusion gate |
| OTR-02 position × layer patching | `scripts/otr02_position_layer_patching.py` | 6 dev | Qwen2.5, OLMo-2, Gemma | Superseded (by OTR-03/09) |
| OTR-03 name-state transplantation | `scripts/otr03_code_transplant.py` | 64 dev (core, BX transitive, BX psych, es, gl; 70B) + 40 test0 | five core + Llama 70B | **Confirmatory** C2, C3 (English 10/10, BX 10/10) and XL-C2, XL-C3 (es 9/10, gl 10/10). **Failed**: Gemma es XL-C3 |
| OTR-04 question-side patching | `scripts/otr04_question_patching.py` | 13 dev | 10 models | Superseded (by OTR-13/14) |
| OTR-05 robustness | `scripts/otr05_robustness.py` | 9 dev | 9 models | Exploratory (not summarised) |
| OTR-06 logit lens | `scripts/otr06_logit_lens.py` | 12 dev | Gemma, Qwen3.8 | Completed (development, historical review package) |
| OTR-07 answer process | `scripts/otr07_answer_process.py` | 10 dev + 3 test0 | five core | **Confirmatory** C7 (3/3 claimed). No claim for phi-4 (dev .29) or OLMo (dev .59) |
| OTR-08 role probes | `scripts/otr08_role_probes.py` | 2 dev | Gemma, Qwen3.8 | Exploratory |
| OTR-09 answer-position and head patching | `scripts/otr09_answer_patching.py` | 69 dev (core, BX ×3, es, gl, UD, WIKI real/neutral; 70B) + 20 test0 | five core + both 70B | **Confirmatory** C4, C5 (English 9/10), BX C5 (5/5), XL-C4/C5/C5b (es 15/15, gl 15/15). **Failed**: phi-4 C5 (`other_failing` control) |
| OTR-10 incremental role probes | `scripts/otr10_incremental_roles.py` | 4 dev | Gemma, Qwen3.8 | Exploratory (not summarised) |
| OTR-11 role steering | `scripts/otr11_role_steering.py` | 2 dev | Gemma, Qwen3.8 | Exploratory (not summarised) |
| OTR-12 knockout and role vector | `scripts/otr12_knockout_and_role_vector.py` | 5 dev | five core | Exploratory, summarised: partial necessity (supplement S3 §5, Interventions §4) |
| OTR-13 question-token patching | `scripts/otr13_question_words.py` | 15 dev (core, second working question, BX) | five core | Completed (development); frozen "by" layers taken from it |
| OTR-14 layer pipeline | `scripts/otr14_layer_pipeline.py` | 10 dev (core, BX) + 5 test0 | five core | **Confirmatory** C6, C8 (10/10); pipeline ordering exploratory |
| OTR-15 prompt formats | `scripts/otr15_prompt_robustness.py` | 5 dev | five core | Exploratory (supplement S2 §3) |
| Model survey (battery) | `scripts/otr_battery.py` | 9 | 5 core + gpt-oss | Completed (development) |
| API behaviour | `scripts/otr_generate_api.py`, `api_behaviour.py` | **none in any branch** | – | **Not available.** The runner and tests exist, but there are no result files, so this is not evidence |

Test0 totals: English 37/38, BX 15/15, Spanish 29/30, Galician 30/30, all reproduced from `main` by the frozen scorer.

## Materials and generators

| Material | Generator (on `main`) | Evidence |
|:--|:--|:--|
| English core | `order_role_stimuli.py` v1.3.0 | Development and test0 |
| BX transitive / ditransitive / psych | `broad_stimuli.py` v0.1.0 | Transitive: test0 (C2, C3, C5). Ditransitive and psych: development only |
| Spanish, Galician | `multilingual_stimuli.py` v1.0.0 | Development and test0 (XL addendum) |
| Stranded-P battery | `stranded_p_stimuli.py` | Development (historical review package) |
| UD natural sentences | `ud_stimuli.py`, `ud_en_natural.json` (third-party, see `THIRD_PARTY_DATA.md`) | Development (OTR-09 transfer) |
| WIKI-ROLES | `wiki_roles.py`, `scripts/wiki_*` | Development (OTR-09 transfer). **Item sets in the archive only** |
| Swedish | `swedish_stimuli.py` SV 0.1.0 | Development pilot. **Native-speaker check not done** |

## Protocols and frozen settings

FREEZE.md, FREEZE_XL.md, `freeze_inputs.sha256`, `test0_attempts.md` (template) and the scorer v1.1.0 are on `main`, unchanged since the freeze commits. All three freeze commits are ancestors of `main`. The canonical attempt log (`results/test0_attempts_pelle.md`) arrives with this PR. All 38 frozen development inputs are on `main` after this PR: 18 were already there, and this PR adds 20.

## Proposed or not done (not presented as completed)

- **Test sets 1–3**: closed, never run.
- **70B freeze addendum**, and **OTR-03 for Qwen2.5 72B**: not run.
- **Native-speaker check of the Swedish stimuli**: pending.
- **Normative Galician variant** (without the prepositional "a"): may be run later as exploratory work (FREEZE_XL.md §6). Not run.
- **Psych-verb class coefficients**: quoted in RESULTS §2 from the 20 November development log. The underlying OTR-03 BX-psych outputs are on `main` (10 files), but the class-level split has not been recomputed by a build script.
- **Numerical summaries of OTR-05, OTR-10 and OTR-11**: the runs exist and are not summarised on the Paper II pages. OTR-12 was summarised on 28 November from its existing outputs.
- **Complete or unique circuit**: not claimed (see RESULTS §9).

## Where each part of the record lives

| Part | Location |
|:--|:--|
| Code, generators, protocols, scorer, tests, CI | `main` (PR #19, with history) |
| Development, test0 and 70B outputs, run logs, plots | `results/` on `main` (PR #12 plus this PR) |
| WIKI-ROLES item sets, check sheets, superseded item set | Archive only (third-party text) |
| Curated results, figures, statistics, supplement | `papers/order-to-role/` |
| Historical review packages and state snapshots | `docs/order_to_role/` (indexed in `supplementary/historical-packages.md`) |
| Superseded versions committed to `order-to-role/stimuli` (e.g. the reverted 16 Nov stranded-question run, 25 blobs) | `main`'s history, through the PR #19 merge |
