[← Learning Relations](README.md) · [Experiment index](EXPERIMENTS.md)

# Evidence and limitations

This page is a **reading aid**, not a replacement for the original protocols or the [working manuscript](../../paper/manuscript.md). Read each result at its recorded evaluation status.

| Question / observation | Evidence to inspect | What it does **not** show |
|:--|:--|:--|
| Does target exposure help within the studied synthetic grammar? | [ML-080](../../reports/ml080/protected_evaluation.json), [ML-084](../../reports/ml084/protected_evaluation.json), [cross-cohort summary](../../reports/paper_cross_cohort/archived_error_taxonomy.md) | General transfer to ordinary language |
| Do joint-role errors behave like individual answer accuracy? | [Manuscript](../../paper/manuscript.md), [cross-cohort audit](../../reports/paper_cross_cohort/archived_error_taxonomy.md) | That every repeated or duplicate answer reflects a single neural failure mode |
| Are role-related states sensitive to intervention? | [ML-085](../../reports/ml085/protected_causal_evaluation.json), [ML-086 interpretation](../../reports/ml086/interpretation.md) | Necessity, mediation of exposure benefits or a unique native role circuit |
| Do proposed mechanistic interactions replicate? | [ML-087](../../reports/ml087/calibration_evaluation.json), [ML-099](../../reports/ml099_role_q/protected_evaluation.json), [ML-100](../../reports/ml100_source_access/exploratory_summary.json) | That a positive exploratory contrast is independently validated; several tested predictions failed |
| Does a structural external benchmark validate the task? | [Manuscript's SLOG discussion](../../paper/manuscript.md) | Protected SLOG structural-generalization performance; the pilot was development-only |

## Newer work outside `main`

The experimental branch has newer work that should be reviewed against its own protocols before paper incorporation:

- [EXP-109 prospective within-grammar replication](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/experiments/exp102-clean-design/paper/EXP109_results_summary_v0.md) — a construction/position contrast, not general linguistic transfer.
- [EXP-110 read-only error decomposition](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/experiments/exp102-clean-design/paper/EXP110_readonly_error_analysis_v0.md) — post-hoc analysis of EXP-109, not an independent replication.
- [PHASE2-03 fixed protocol](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/experiments/exp102-clean-design/paper/PHASE2_03_distractor_event_binding_v0_protocol.md) — a specified within-project distractor sensitivity test; inspect the archived results and analysis before interpreting it.

These are explicitly **branch-only references**. Their presence here does not mean that all materials have been merged or that the canonical manuscript has adopted their conclusions.

## Interpretive boundaries

- Synthetic training supports control, not ecological validity by default.
- Independent training seeds do not create independent linguistic task families.
- Restricted-candidate accuracy must be distinguished from native full-vocabulary responses.
- Layerwise decodability and intervention sensitivity are narrower than a demonstrated native pathway.
- Failed gates and inconclusive replications remain visible.

Original evaluation details and limitations: [scratch checkpoint](../../investigations/scratch_transformers/CURRENT_FINDINGS.md) · [evidence policy](../../docs/EVIDENCE_POLICY.md).
