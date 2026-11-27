[← Order-to-Role](README.md) · [Methods](METHODS.md) · [Results](RESULTS.md)

# Experiment map

Each OTR experiment, the question it answers, its evidence status and where its result is discussed. The code links point to the canonical files on `main`, which arrived with their full history from `order-to-role/stimuli` (PR #19). Each linked file is byte-identical to the frozen code snapshot `9c43792` that the rest of this section cites. All result files are in `results/` on `main` (see the [results archive inventory](archive/README.md)).

| Study | Question | Status | Result | Code |
|:--|:--|:--|:--|:--|
| OTR-01 | Behavioural gate: which models answer reliably? | Development | Qwen2.5 fails; Gemma passes except one wording | [otr01](../../scripts/otr01_behavior_gate.py) |
| Generation check | Do generated answers agree with first-token scores? | Development; **C1, XL-C1 [test0]** | Forced choice misrepresented chat models; failing question fails selectively | [otr_generate_check](../../scripts/otr_generate_check.py) |
| OTR-02 | Where do position × layer patches change the answer? | Development | Answer read late from the name position (Qwen2.5) | [otr02](../../scripts/otr02_position_layer_patching.py) |
| OTR-03 | Do name states carry order, function or role? | Development; **C2, C3 [test0]** | Role largest, mid-network ([Results §2](RESULTS.md#2-the-names-carry-their-thematic-role)) | [otr03](../../scripts/otr03_code_transplant.py) |
| OTR-04 | Which question positions influence the answer? | Development | Superseded by OTR-13/14 | [otr04](../../scripts/otr04_question_patching.py) |
| OTR-05 | Robustness to perturbations | Development | – | [otr05](../../scripts/otr05_robustness.py) |
| OTR-06 | How do name preferences evolve with depth (logit lens)? | Development | Review package, figure f4 | [otr06](../../scripts/otr06_logit_lens.py) |
| OTR-07 | Does the model know the agent when it answers wrong? | Development; **C7 [test0]** | Yes in Gemma, Qwen3.8, Granite ([Interventions §6](INTERVENTIONS.md#6-the-answer-process-does-the-model-know-otr-07)) | [otr07](../../scripts/otr07_answer_process.py) |
| OTR-08 | Where can role be decoded (probes)? | Development | The subject is readable at the full stop, but editing that state does not move the answer (with OTR-10/11) | [otr08](../../scripts/otr08_role_probes.py) |
| OTR-09 | Can answer-position or head patching repair the failure? | Development; **C4, C5, XL-C4/5/5b [test0]** | Handover layer; 20 heads ([Results §3–4](RESULTS.md#3-from-storage-to-answer-the-layer-pipeline)) | [otr09](../../scripts/otr09_answer_patching.py) |
| OTR-10 | How does role information arise token by token? | Development | – | [otr10](../../scripts/otr10_incremental_roles.py) |
| OTR-11 | Can a role direction steer the answer? | Development | – | [otr11](../../scripts/otr11_role_steering.py) |
| OTR-12 | Are the heads necessary; does an averaged role vector transfer? | Development | Heads needed for 51–82% of the agent margin in four models (16% in OLMo), not for the answer; no single steering direction ([Interventions §4](INTERVENTIONS.md#4-knockout-and-steering-otr-12)) | [otr12](../../scripts/otr12_knockout_and_role_vector.py) |
| OTR-13 | Which question token carries the failure? | Development; frozen "by" layers | "by" ([Layers §2](LAYERS.md#2-stage-two-the-question-and-the-word-by-otr-13-otr-14)) | [otr13](../../scripts/otr13_question_words.py) |
| OTR-14 | When are question and names read; how do the curves relate? | Development (pipeline exploratory); **C6, C8 [test0]** | Question-to-answer handover ([Layers §3](LAYERS.md#3-stage-three-the-handover-to-the-answer-position-otr-09-otr-14)) | [otr14](../../scripts/otr14_layer_pipeline.py) |
| OTR-15 | Does the generation effect survive prompt-format changes? | Development | Exploratory | [otr15](../../scripts/otr15_prompt_robustness.py) |

A dash in the Result column means the run exists but is not summarised on these pages.

## Scoring and provenance

- [Frozen scorer `otr_freeze_score.py`](../../scripts/otr_freeze_score.py) v1.1.0. English rules unchanged since v1.0.0; Spanish/Galician added in v1.1.0.
- [English freeze](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE.md), [Spanish/Galician addendum](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE_XL.md), [freeze input checksums](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/freeze_inputs.sha256).
- [25 November result index](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/state_2026-11-25/result_index.md) and [development results log](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/state_2026-11-25/results_log.md).
- [November review package](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/review_2026-11/README.md): the forced-choice correction and early figures.

## Stimulus generators

| Generator | Material |
|:--|:--|
| [`order_role_stimuli.py`](../../src/relational_lab/experiments/order_role_stimuli.py) | Core constructions, splits |
| [`stranded_p_stimuli.py`](../../src/relational_lab/experiments/stranded_p_stimuli.py) | Stranded-preposition question battery |
| [`broad_stimuli.py`](../../src/relational_lab/experiments/broad_stimuli.py) | BX: transitive, ditransitive, psych |
| [`multilingual_stimuli.py`](../../src/relational_lab/experiments/multilingual_stimuli.py) | Spanish, Galician |
| [`ud_stimuli.py`](../../src/relational_lab/experiments/ud_stimuli.py), [`ud_extract_natural.py`](../../scripts/ud_extract_natural.py) | Universal Dependencies sentences |
| [`wiki_roles.py`](../../src/relational_lab/experiments/wiki_roles.py), [`wiki_extract_roles.py`](../../scripts/wiki_extract_roles.py) | WIKI-ROLES |
| [`swedish_stimuli.py`](../../src/relational_lab/experiments/swedish_stimuli.py) | Swedish (not yet native-checked) |
