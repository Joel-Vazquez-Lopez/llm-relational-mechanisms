[← Research guide](../STUDIES.md)

# Repository map and research ownership

The repository has **two paper-facing paths** under `papers/` and one shared technical codebase. This is a navigation layer: canonical scripts, protected protocols and existing result paths are not moved by this documentation change.

| Entry point | Purpose |
|:--|:--|
| [Learning Relations](learning-relations/README.md) | Scratch-trained paper, evidence, experiment index, reproducibility |
| [Order-to-Role](order-to-role/README.md) | Pretrained-LLM paper, evidence, experiment index, layer plan and reproducibility |
| [Scratch manuscript](../paper/manuscript.md) | Canonical editable working manuscript |
| [Scratch investigation](../investigations/scratch_transformers/README.md) | Historical research narrative and checkpoints |
| [Earlier pretrained investigations](../investigations/pretrained_llms/README.md) | Related ML studies, not an automatic part of OTR |
| [Reports](../reports/README.md) | Protocols, summaries, analyses and references to underlying data |
| [Shared standards](../docs/EVIDENCE_POLICY.md) | Evidence classification, data stewardship and scientific provenance |

## Where files actually live

| Canonical path | Responsibility / note |
|:--|:--|
| `paper/` | Scratch manuscript and historical ML/EXP protocols; preserve frozen paths |
| `scripts/ml*.py`, `scripts/exp*.py` | Existing experiment entrypoints; **prefix alone does not decide paper ownership** |
| `scripts/otr*.py` | Order-to-Role entrypoints, currently on the [stimuli branch](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/stimuli/scripts) |
| `src/relational_lab/` | Shared package and instrumentation; OTR implementations live on its code branch |
| `investigations/` | Historical track-level overviews and indexes |
| `reports/` | Archived experiments, protocols, evidence and analysis summaries |
| `results/` | Often local/gitignored outputs; some historical tracked outputs and a separate OTR archive |
| `docs/` | Shared reproducibility policies; OTR review figures currently on the stimuli branch |
| `tests/` | Root tests and branch-specific OTR tests |

## Current branches and integration boundaries

| Source | Intended treatment |
|:--|:--|
| `main` | Public navigation and established canonical material |
| [`experiments/exp102-clean-design`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/experiments/exp102-clean-design) | Review post-merge EXP-109–116 and PHASE2 additions independently |
| [`order-to-role/stimuli`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/stimuli) | Integrate OTR implementations, lightweight tests and documentation selectively |
| [`order-to-role/results-dev`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/results-dev) | Preserve development results for provenance; do not blindly copy all outputs to `main` |
| Other historical branches | Inspect whether they contain unique work before closing or archiving |

Use the [integration plan](../docs/INTEGRATION_PLAN.md) for sequencing and preservation checks.

## Principles

1. Do not change frozen script bytes, paths, evaluation splits or protocol hashes merely to make folders look symmetrical.
2. Keep manuscript claims separate from exploratory observations, selected interventions and independent replications.
3. Record model revisions, generator versions, seed and split provenance, scoring rules and rights status when publishing results.
4. Keep branch-only files explicitly labeled until code and reports are actually integrated.
5. Do not automatically assign historical `ML-` studies to either paper based on a filename prefix.

**Documentation map, not a certification that all archived results are reproducible from `main`.**
