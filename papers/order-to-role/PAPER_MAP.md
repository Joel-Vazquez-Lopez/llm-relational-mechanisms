[← Order-to-Role](README.md) · [Layer investigation](LAYERS.md) · [Evidence](EVIDENCE.md)

# Paper architecture

A reading guide to the [24 November manuscript draft (25 November state snapshot)](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/state_2026-11-25/paper_draft.md), not a new manuscript or a change to the frozen claims. The live manuscript remains the author's canonical writing workspace.

## Proposed main-paper sequence

| Section | Scientific purpose | Experiments / evidence |
|:--|:--|:--|
| 1. Paradigm and measurement | Decorrelate order, grammatical function and thematic role; establish scoring validity | OTR-01, generation check, OTR-05/15; C1 |
| 2. Participant representations | Test whether name-state transplants carry role-specific effects beyond order and function | OTR-03, OTR-08/10; C2/C3; psych-verb development contrasts |
| 3. Computation across layers | Separate role-related name states, question-side effects and answer-position readout | OTR-02/04/06/13/14; C4/C6/C8 |
| 4. Causal readout and failure | Test frozen head-set repair, matched controls, and neutral-prefix answer effects | OTR-07/09/11/12; C5/C7 |
| 5. Generality | Transfer English-selected heads to new constructions, Spanish/Galician and natural sentences | OTR-03/09, BX, XL, UD, WIKI; [transfer map](TRANSFER.md) |
| 6. Frozen evaluation | Report every prespecified model/claim result, failures and not-evaluable cells | [English freeze](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE.md); [XL addendum](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/FREEZE_XL.md) |
| 7. Discussion | Distinguish causal evidence from a unique native pathway; discuss linguistic, measurement and selection limits | [Evidence ledger](EVIDENCE.md) |

The existing draft already uses a related sequence. This table groups it for navigation, not to mandate a change in paper numbering.

## Proposed figure order (not generated here)

| Figure | Purpose | Primary source |
|:--|:--|:--|
| 1 | Controlled linguistic design, matched behavioral contrast and generation validity | OTR-01 and generation runs |
| 2 | Role, function and position transplant coefficients across depth; psych predicates | OTR-03 |
| 3 | Layer pipeline: question states, attention blocking, answer-state repair | OTR-13/14; [layer guide](LAYERS.md) |
| 4 | Frozen-head repair versus layer-matched and content controls | OTR-09; per-item uncertainty |
| 5 | Transfer across BX, Spanish, Galician, UD and Wikipedia; separate development and test0 | [Transfer guide](TRANSFER.md) |
| 6 or appendix | Complete frozen claim matrix, unsuccessful tests, sensitivity and limitations | Frozen scorer and result provenance |

This is a proposed figure map, not a statement that all contrasts passed. Report original margins, effect sizes, denominators and intervals rather than mixing repair ratios with generation accuracy.

## Narrative boundaries

The central hypothesis concerns role-sensitive representation and question-conditioned selection. A coherent set of interventions supports studying that hypothesis, but temporal ordering is not a causal chain by itself. Preserve the earlier forced-choice artifact correction, distinguish the conditional repair estimand from overall accuracy, and distinguish validated transfer from exploratory results. Cross-language generality does not establish language invariance of every component or representation.

For anonymous submission, prepare a separate anonymized manuscript/artifact. The public repository guide is not a submission artifact.
