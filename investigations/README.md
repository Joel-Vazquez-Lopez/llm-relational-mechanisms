[← Model Linguistics](../README.md)

# Historical Development

## From an initial question to two research papers

Model Linguistics began as a broader investigation of how Transformers represent and use linguistic relationships. Early studies tested participant selection, event binding, question-dependent retrieval, and causal changes to model states. The programme developed along two increasingly distinct lines: experiments on pretrained LLMs and controlled learning experiments with small Transformers trained from scratch.

This page is the **historical entrance**: a guide to the sequence of research questions, experimental decisions, failures, and the work that motivated the paper directions. It is not a third paper, and it does not classify every historical result as evidence for either current manuscript.

## Research chronology

| Phase | Historical programme | What was investigated | Where to read |
|:--|:--|:--|:--|
| Foundation | October 2026; official repository established | Initial candidate-versus-global output diagnostic and research standards | [Research log](../docs/RESEARCH_LOG.md), [scientific charter](../docs/RESEARCH_CHARTER.md) |
| Early pretrained exploration | ML-001–036 | Role selection, question and predicate effects, counterfactual binding, layer mapping, event and answer disentanglement | [Research direction](../reports/2026-10-26_cycle2/RESEARCH_DIRECTION.md), [ML-026–029](../reports/cycle3/ML026_ML029_CHECKPOINT.md), [ML-032–036](../reports/cycle3/ML032_ML036_CHECKPOINT.md) |
| Controlled-model pilots | ML-037–047 | Scratch training, factorial interventions, transfer and causal role routing | [Scratch investigation](scratch_transformers/README.md) |
| Pretrained follow-ups | ML-048–055 | Query interaction, position counterfactuals, factor models and measurement audits | [Pretrained experiment index](pretrained_llms/EXPERIMENT_INDEX.md) |
| Learning and developmental studies | ML-056–087 | Controlled exposure, checkpoints, role-error taxonomy, generalization and causal sensitivity | [ML-056 report](../reports/ml056/README.md), [scratch findings](scratch_transformers/CURRENT_FINDINGS.md), [scratch experiment index](scratch_transformers/EXPERIMENT_INDEX.md) |
| Later ML studies and transition | ML-088–100 | Probe access, routing, source-state and question-side interventions, prospective replication and bounded screens | [Complete ML-series index](LEGACY_INDEX.md), [report archive](../reports/README.md) |

The ranges are a reading sequence for the archived ML programme, **not a claim that experiments were conducted strictly in numerical order or that every ML-088–100 study predates a single verified separation date**. Some historical studies also contribute evidence to a current paper. A paper's own evidence ledger determines whether and how that work is cited.

## Explore the complete archived ML series

**[Full historical experiment index (ML-001–100)](LEGACY_INDEX.md)**

The index enumerates all 109 matching Python entrypoints in the checked-in ML series at this snapshot. Multiple entrypoints can belong to one experiment; missing numbers and local-only artifacts are not inferred from filenames.

For scientific context and original reporting, use the existing [pretrained research history](pretrained_llms/README.md) and [scratch research history](scratch_transformers/README.md). Original execution paths, reports, failed gates and protocol versions remain canonical.

## The two paper directions

| Paper | Research question |
|:--|:--|
| **[Paper I — Relational Learning in Transformers](../papers/learning-relations/README.md)** | How controlled experience shapes relational abilities, errors and mechanisms during training. |
| **[Paper II — Order-to-Role](../papers/order-to-role/README.md)** | How pretrained models represent participant roles and select answers under different linguistic conditions. |

Newer paper-specific EXP, PHASE2 and OTR experiments belong in the corresponding paper's experiment map, not in the shared historical chronology.

## Research infrastructure

[Shared infrastructure](shared/README.md) · [Evidence policy](../docs/EVIDENCE_POLICY.md) · [Provenance](../docs/PROVENANCE.md) · [Repository map](../papers/MAP.md) · [Detailed study guide](../STUDIES.md)

The historical record is preserved even when a hypothesis fails or the experimental programme changes direction. This documentation does not relocate scripts, data, checkpoints or frozen protocols.
