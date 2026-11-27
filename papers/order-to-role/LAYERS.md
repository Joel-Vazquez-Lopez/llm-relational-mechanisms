[← Order-to-Role](README.md) · [Paper architecture](PAPER_MAP.md)

# Layer investigation: participant states to answers

The candidate computational account has three separable parts: role-related information associated with participant tokens, question-conditioned selection, and answer-position readout. OTR-14 puts multiple intervention curves on a common per-item scale.

## 1. Participant representation

| Measurement | Entry point | Interpretation |
|:--|:--|:--|
| Single-name causal transplants | [OTR-03](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/scripts/otr03_code_transplant.py) | Role, function and order effects on recipient-name preference |
| Probes and incremental token states | [OTR-08](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/scripts/otr08_role_probes.py), [OTR-10](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/scripts/otr10_incremental_roles.py) | Decodability and emergence, not proof of causal use |
| Role steering | [OTR-11](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/scripts/otr11_role_steering.py) | Sensitivity to a selected direction |

## 2. Question-side computation

| Measurement | Entry point |
|:--|:--|
| Position and question-state patching | [OTR-02](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/scripts/otr02_position_layer_patching.py), [OTR-04](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/scripts/otr04_question_patching.py) |
| Interchange, mean ablation and reverse damage at question words | [OTR-13](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/scripts/otr13_question_words.py) |
| Question/name attention blocking, joint-question and answer-state swaps, failing/patient `by` transfers | [OTR-14](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/scripts/otr14_layer_pipeline.py) |

OTR-14 reports family-bootstrap intervals and checks whether unmodified attention-mask hooks reproduce clean logits. Architectures without the necessary standard-attention interface may lack some curves; report missing data explicitly.

## 3. Answer selection

| Measurement | Entry point |
|:--|:--|
| Intermediate name preferences | [OTR-06](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/scripts/otr06_logit_lens.py) |
| Answer generation and neutral by-slot prefix | [OTR-07](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/scripts/otr07_answer_process.py) |
| Residual repair, frozen head sets, control patches | [OTR-09](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/scripts/otr09_answer_patching.py) |
| Head knockout and averaged role vector | [OTR-12](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/scripts/otr12_knockout_and_role_vector.py) |

## 4. Frozen layer hypotheses

The following indices are **development-selected settings**, transcribed from the [English freeze, section 4](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE.md). They are physical layer indices, not post-test discoveries.

| Model | Layers | Role layer | Question `by` layer | Answer handover |
|:--|--:|--:|--:|--:|
| Gemma 4 31B | 60 | 28 | 28 | 40 |
| Qwen 3.8 27B | 64 | 29 | 30 | 54 |
| Phi-4 | 40 | 15 | 8 | 22 |
| OLMo 3.1 32B | 64 | 18 | 8 | 26 |
| Granite 4.2 30B | 64 | 37 | 28 | 52 |

Role layer, question-token layer and handover are defined by different measurements. Their order differs by model; do not force a universal serial order. The XL addendum freezes language-specific handover layers. OTR-14 pipeline ordering remains exploratory under the English freeze, even though specific token interventions are tested.

## 5. What would establish a stronger pathway account?

Report paired item/family uncertainty, split-specific donors, token alignment, source-content and layer-matched controls, selective disruption and restoration, model-specific exceptions, and behavior in generated answers. Frozen definitions govern the confirmatory tests; additional analyses remain exploratory. See [evidence and validation status](EVIDENCE.md).
