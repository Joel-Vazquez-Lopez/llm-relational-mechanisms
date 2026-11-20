[← Order-to-Role](README.md) · [Evidence](EVIDENCE.md)

# Order-to-Role experiment map

Active scripts reside on **[`order-to-role/stimuli`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/stimuli/scripts)**, not on `main`. Development output files primarily reside on **[`order-to-role/results-dev`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/results-dev/results)**. This map names research tools; it does **not** imply that each proposed hypothesis passed its test.

| Study | Main question | Code on stimuli branch |
|:--|:--|:--|
| OTR-01 | Behavioral scoring and competence gates | [Behavioral gate](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr01_behavior_gate.py) |
| OTR-02 | Where do position/layer patches change outputs? | [Position × layer patching](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr02_position_layer_patching.py) |
| OTR-03 | How portable are name/role-related states? | [Code transplantation](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr03_code_transplant.py) |
| OTR-04 | Which question-side positions influence answer selection? | [Question patching](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr04_question_patching.py) |
| OTR-05 | Which effects persist under perturbations? | [Robustness](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr05_robustness.py) |
| OTR-06 | How do name preferences evolve with depth? | [Logit lens](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr06_logit_lens.py) |
| OTR-07 | What changes during answer processing? | [Answer process](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr07_answer_process.py) |
| OTR-08 | Where can role information be decoded? | [Role probes](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr08_role_probes.py) |
| OTR-09 | Can answer-position patching repair failures? | [Answer patching](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr09_answer_patching.py) |
| OTR-10 | How does role information arise during incremental processing? | [Incremental roles](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr10_incremental_roles.py) |
| OTR-11 | Can role-directed intervention steer answers selectively? | [Role steering](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr11_role_steering.py) |
| OTR-12 | Which head contributions are necessary, and can role vectors transfer? | [Head knockout and role vector](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr12_knockout_and_role_vector.py) |

## Stimulus families and measurement tools

- [Original order/subject/role contrasts](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/src/relational_lab/experiments/order_role_stimuli.py) — actives, passives, clefts and held-out relatives.
- [Broad English constructions](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/src/relational_lab/experiments/broad_stimuli.py) — new names/verbs, transitive, ditransitive and psych predicates.
- [Swedish stimuli](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/src/relational_lab/experiments/swedish_stimuli.py) — marked as **not yet native-checked** in its generation notes.
- [Generation and answer extraction](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr_generate_check.py) — important for separating prompt/format effects from candidate-restricted scoring.
- [November reviewer package](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/docs/order_to_role/review_2026-11/README.md) — tables, figures and versioned provenance.

## Reading order

Begin with [behavior and scoring validity](EVIDENCE.md). Then follow [layerwise information versus influence](LAYERS.md). Inspect the [raw development results](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/results-dev/results) only alongside their generating code and manifests. Do not infer that a model has a grammatical mechanism solely because a head or residual state changes a response.
