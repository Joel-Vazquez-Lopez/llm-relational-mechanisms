[← Two studies](../../STUDIES.md) · [Companion study](../learning-relations/README.md)

# Order-to-Role

### Grammatical roles, response conditions and answer selection in pretrained language models

**Model class:** pretrained LLMs  
**Stage:** exploratory, development-stage investigation  
**Core question:** how do models distinguish linear order, grammatical function and semantic role—and what computations influence which participant becomes the answer?

The controlled stimuli deliberately separate first-mentioned position, syntactic subjecthood and thematic role. The project compares answer formats and question constructions, then examines representations and causal interventions across model layers.

## Read the study

| Purpose | Document |
|:--|:--|
| **What is actually supported?** | [Evidence and limitations](EVIDENCE.md) |
| **What was tested?** | [OTR-01–12 experiment index](EXPERIMENTS.md) |
| **What happens across layers?** | [Representations to answers](LAYERS.md) |
| **How to inspect code and outputs?** | [Reproducibility and data guide](REPRODUCE.md) |
| **External November review** | [Review package](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/docs/order_to_role/review_2026-11/README.md) |
| **Critical questions** | [Reviewer brief](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/order-to-role/stimuli/docs/order_to_role/review_2026-11/REVIEW_BRIEF.md) |

## Three questions, three kinds of evidence

**Behavioral measurement.** Does first-token candidate scoring measure the same response as generated names or full sentences? Are apparent errors affected by instruction wording and answer format?

**Linguistic structure.** Which effects persist across actives, passives, clefts and newer vocabulary or grammatical families? Unusual questions and templated forms must not be silently treated as natural-language competence tests.

**Internal computation.** Where is participant/role information accessible? Which changes influence the final selected participant? Probes, logit lenses, attention and MLP interventions answer related but **distinct** questions.

## Status and scope

The November development-only review reports both positive observations and important corrections. Several broad forced-choice error claims were dropped after generation checks exposed scoring artifacts. Within-item response differences are interesting, but do not by themselves prove that role knowledge is present and merely fails to be retrieved. The held-out test sets described in that review were unopened at the time of the review; inspect later experiment provenance before making any newer claim.

**Current work is split across branches:** [code/stimuli](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/stimuli) and [development results](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/results-dev/results). These are **not yet integrated into `main`**.

**Next:** [Evidence](EVIDENCE.md) · [Experiments](EXPERIMENTS.md) · [Layer study](LAYERS.md) · [Reproduce](REPRODUCE.md).
