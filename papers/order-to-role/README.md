[← Research atlas](../../STUDIES.md) · [Other paper →](../learning-relations/README.md)

# 02 / Order-to-Role

> How do word order, grammatical function, semantic role and response conditions shape the answer an LLM selects?

**Pretrained LLMs** · **OTR** · **Active / exploratory**

| Read | Inspect | Explore |
|:--|:--|:--|
| [Review package](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/docs/order_to_role/review_2026-11/README.md) | [Reviewer questions](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/docs/order_to_role/review_2026-11/REVIEW_BRIEF.md) | [Layer study →](LAYERS.md) |
| [Historical pretrained track](../../investigations/pretrained_llms/README.md) | [Data branch](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/results-dev/results) | [Code and tests](../MAP.md) |

## The paper

Working direction: **relational answering across questions, response formats and model layers**.
The final title and claim remain open. The review package targets early-2027 ARR;
this is a research direction, not a publication-readiness assessment.

The core puzzle is a within-item difference between response conditions. Correct
sentence responses alongside wrong bare-name responses may reflect different question
interpretations, extra computation, or different use of relational information.
“Knows the relation but cannot retrieve it” remains a hypothesis requiring controls.

## Three connected parts

| Part | Question | Evidence requirement |
|:--|:--|:--|
| Measurement | Are forced-choice and generated answers measuring the same behavior? | Matched items, validated extraction and explicit answer-format controls |
| Linguistic behavior | Which constructions and questions change participant selection? | Lexical and template holdouts, family-level uncertainty, counterexamples |
| Layerwise explanation | Which internal changes cause the response difference? | Controlled interventions and independent prediction; see the [layer plan](LAYERS.md) |

## Existing tools, not automatically established findings

The branch contains OTR-01–06 behavior, patching, transplant, question, robustness and
logit-lens tools. Newer OTR-07–11 cover answer processing, role probes, answer patching,
incremental roles and steering. Their existence does not establish that the associated
hypotheses passed; inspect each version and its data provenance.

## Next decisions

1. Audit response extraction and reproduce matched-item behavioral contrasts.
2. Separate instruction, answer position and generated intermediate-content effects.
3. Use a discovery split to identify candidate layer/site pathways.
4. Freeze the primary contrast and intervention controls before independent confirmation.
5. Compare model families without treating equal layer numbers as functional alignment.

## Paper figure sequence

1. Controlled constructions and matched-item response differences.
2. Behavioral effects by question, construction and response condition.
3. Layerwise information and intervention maps.
4. Selective disruption/repair and held-out predictions, including failures.

All new figures above are plans. Existing review figures retain their original provenance.
