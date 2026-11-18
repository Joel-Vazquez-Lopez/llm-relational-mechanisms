[← Research atlas](../../STUDIES.md) · [Other paper →](../order-to-role/README.md)

# 01 / Learning relations

> How does controlled experience change relational generalisation and the errors a Transformer makes?

**Scratch-trained Transformers** · **ML / EXP lineage** · **Existing assets frozen for preservation**

| Read | Inspect | Plan |
|:--|:--|:--|
| [Working manuscript](../../paper/manuscript.md) | [Evidence checkpoint](../../investigations/scratch_transformers/CURRENT_FINDINGS.md) | [Next-study outline](#next-study-outline) |
| [Research overview](../../investigations/scratch_transformers/README.md) | [Experiment index](../../investigations/scratch_transformers/EXPERIMENT_INDEX.md) | [Shared repository map](../MAP.md) |

## The paper

*Targeted Experience, Relational Generalization, and Role-Assignment Errors in Transformers.*
The existing working manuscript has a 15 December 2026 deadline. This hub does not
replace its text, protocols or evidence record.

The central behavioral question is how marginal accuracy, joint correctness and
specific error patterns respond differently to training experience. Cohort replication
within one synthetic grammar is distinct from transfer to new linguistic structures.

## Evidence → interpretation → next test

| Evidence entry point | Supported scope | Next question |
|:--|:--|:--|
| [Cross-cohort audit](../../reports/paper_cross_cohort/archived_error_taxonomy.md) | Repeated exposure benefit and joint-error changes within the studied family | Which properties of experience explain the changes? |
| [EXP-104 matched-source audit](../../reports/exp104/analysis/source_matched_dev_3200_audit.md) | No consistent final quartet advantage under this control | Is any benefit primarily about learning speed? |
| [EXP-106 protocol](../../paper/EXP106_voice_recombination_protocol_v0.md) | Exploratory design for direct versus indirect voice exposure | Can the unseen voice combination be solved reliably across seeds? |

The existing manuscript records failed mechanistic predictions and a deferred external
validation. Those remain part of the paper's evidential limits.

## Next-study outline

**Proposed, not frozen and not an instruction to launch experiments.**

1. Audit execution variability and select one interpretable exposure contrast.
2. Freeze evaluation, training budget, seed-level analysis and stopping rules.
3. Track native individual and joint answers, event errors, reversals and duplicate
   answers at fixed checkpoints. Keep oracle-restricted diagnostics separate.
4. Map held-out role/position decoding and controlled intervention effects across
   training time and depth. Decodability does not establish causal use.
5. Freeze one computational prediction before testing new seeds and held-out structures.

## Paper figure sequence

| Figure | Purpose |
|:--|:--|
| 1 · Experience and transfer | Show the training contrast and every seed's generalisation outcome |
| 2 · Errors through learning | Separate accuracy improvement from changes in joint-role errors |
| 3 · Time × layer | Compare accessible information with intervention sensitivity |
| 4 · Independent prediction | Test where a proposed explanation transfers or fails |

These are figure plans; no new results or completed mechanism claims are implied.
