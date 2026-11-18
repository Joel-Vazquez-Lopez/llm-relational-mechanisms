[← OTR paper hub](README.md) · [Research atlas](../../STUDIES.md)

# Inside the answer / layer-study map

**Question:** when response conditions change participant selection, which internal
computations change, and can interventions predictably change the outcome?

**Status:** proposed synthesis of the study direction, not a frozen protocol.

## Follow three signals

| Signal | Measurement | Interpretation limit |
|:--|:--|:--|
| Role, position and participant information | Held-out probes across names, verbs and constructions | Accessible to a decoder does not mean used by the model |
| Answer preference | Logit lens and native output measurements | Intermediate readout is not causal localization |
| Influence on selection | Counterfactual state, component and path interventions | Sensitivity alone does not identify a unique native mechanism |

## From a map to an explanation

**A · Map depth.** Record participant, predicate, question and answer-position states
across all layers on matched items. Keep consistent successes, failures and format-sensitive
items visible; avoid selecting favorable cases from the final evaluation.

**B · Locate influence.** Patch role-swapped and construction-matched donors. Include
same-answer/different-relation and different-answer/same-role controls. State how spans
are aligned when token lengths differ; never assume raw token offsets are equivalent.

**C · Trace a candidate path.** Separate attention and MLP updates at candidate sites,
then test sender-to-receiver paths. Query/key/value analysis is a follow-up where a
specific selection-versus-content hypothesis warrants it, not a mandatory sweep.

**D · Predict and confirm.** Freeze sites, controls, primary effects and evaluation
before testing new lexical/template families. Evaluate disruption, restoration and
specificity together. Report contradictory cases and architectural differences.

## The response-format control

Sentence answers may name the participant after additional generated tokens. Separate
instruction wording, answer location and intermediate generated context before interpreting
format differences as a retrieval failure. Compare explicitly defined readout positions;
do not equate a first-token name score with full generated-answer correctness.

## Script entry points on the OTR branch

| Tool | Entry point |
|:--|:--|
| Position × layer patching | [OTR-02](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr02_position_layer_patching.py) |
| Code transplantation | [OTR-03](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr03_code_transplant.py) |
| Question patching | [OTR-04](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr04_question_patching.py) |
| Logit lens | [OTR-06](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr06_logit_lens.py) |
| Answer processing | [OTR-07](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr07_answer_process.py) |
| Role probes | [OTR-08](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr08_role_probes.py) |
| Answer patching | [OTR-09](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr09_answer_patching.py) |
| Incremental roles | [OTR-10](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr10_incremental_roles.py) |
| Role steering | [OTR-11](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/scripts/otr11_role_steering.py) |

## Before calling an explanation supported

- Measurement and extraction validated on development data.
- Matched donors, self-patches, bidirectional swaps and perturbation checks reported.
- Primary effect distinguishes role information from position and answer identity.
- Confidence intervals respect the underlying stimulus-family dependence.
- Selected explanation predicts a fresh evaluation without selecting a new best layer.

This checklist is a planning aid. Existing frozen protocol requirements take precedence.
