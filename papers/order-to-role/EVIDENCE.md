[← Order-to-Role](README.md) · [Experiments](EXPERIMENTS.md)

# Evidence, corrections and limitations

This is a **navigation and interpretation ledger**, not a new statistical analysis. The source is the [November development-only review](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/docs/order_to_role/review_2026-11/README.md). Consult that package's provenance table and original model outputs before promoting a result to the manuscript.

| Research observation | Review status / qualification | Primary source |
|:--|:--|:--|
| Apparent object-question failures under first-token forced choice | **Scoring artifact for several chat models.** A broad subject/object retrieval-failure claim was dropped; original and generated samples also differ in some comparisons. | [C1, review report](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/docs/order_to_role/review_2026-11/README.md) |
| Strong variation for the marked question “Who was someone V-ed by?” | **Development effect under unusual wording.** Must distinguish question interpretation from grammatical role retrieval. | [C2, review report](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/docs/order_to_role/review_2026-11/README.md) |
| Different answers for bare-name versus full-sentence instructions on matched items | **Development observation.** Format, instruction interpretation and intermediate computation remain confounded without more controls. | [C3, review report](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/docs/order_to_role/review_2026-11/README.md) |
| Layerwise answer preference | **Descriptive.** A logit-lens readout at intermediate depth is not a causal trace or proof of absence of role information. | [C4, review report](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/docs/order_to_role/review_2026-11/README.md) |
| Role and question-side causal measurements | **Mixed provenance.** Some earlier model comparisons used contaminated forced-choice prompts and need repeating. | [C5, review report](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/docs/order_to_role/review_2026-11/README.md) |
| Stranding and general voice mismatch as explanations | **Counterevidence.** Several simple broad explanations were rejected or narrowed. | [C6–C7, review report](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/docs/order_to_role/review_2026-11/README.md) |

## Newer research directions

The [OTR code branch](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/stimuli) includes subsequent work on answer-state patching, head knockout/role vectors, multilingual stimuli and broader lexical/construction transfer. Its [development results branch](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/results-dev) has later measurements.

These later experiments **exist**, but this navigation page does not certify their primary outcomes, independent validation, absence of test leakage or consistency with the earlier review. Audit the results against the exact code, prompts and split manifests.

## Stronger claims require stronger controls

- Compare genuine generated answers, name-only answers and candidate-restricted scores on the same items.
- Test conventional question formulations alongside marked or awkward formulations; do not conflate them.
- Cluster uncertainty by independent stimulus family rather than counting paraphrases as independent samples.
- Validate tokenization, answer extraction, prompt templates and source/target alignment across architectures.
- Prespecify selected components, negative controls and held-out constructions before a prospective mechanistic prediction.

See [the layer-plan interpretation boundaries](LAYERS.md), [experiment index](EXPERIMENTS.md) and [evidence policy](../../docs/EVIDENCE_POLICY.md).
