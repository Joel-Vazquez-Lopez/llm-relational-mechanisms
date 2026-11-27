[← Research guide](../STUDIES.md)

# Repository map and research ownership

The repository has **two paper-facing paths** under `papers/` and one shared technical codebase. This is a navigation layer: canonical scripts, protected protocols and existing result paths are not moved by this documentation change.

| Entry point | Purpose |
|:--|:--|
| [Historical Development](../investigations/README.md) | Original research chronology and complete tracked ML-series index |
| [Paper I — Relational Learning in Transformers](learning-relations/README.md) | Scratch learning emergence, divergent trajectories, selective collapse, mechanism/repair and origins study; evidence and reproducibility |
| [Paper II — Order-to-Role](order-to-role/README.md) | Pretrained-LLM paper architecture, evidence, experiment index, layer investigation, transfer and reproducibility |
| [Scratch manuscript](../paper/manuscript.md) | Canonical editable working manuscript |
| [Scratch investigation](../investigations/scratch_transformers/README.md) | Historical research narrative and checkpoints |
| [Earlier pretrained investigations](../investigations/pretrained_llms/README.md) | Related ML studies, not an automatic part of OTR |
| [Reports](../reports/README.md) | Protocols, summaries, analyses and references to underlying data |
| [Shared standards](../docs/EVIDENCE_POLICY.md) | Evidence classification, data stewardship and scientific provenance |

## Where files actually live

| Canonical path | Responsibility / note |
|:--|:--|
| `paper/` | Scratch manuscript and historical ML/EXP plus newer branch-specific protocols; preserve frozen paths |
| `scripts/ml*.py`, `scripts/exp*.py` | Existing experiment entrypoints; **prefix alone does not decide paper ownership** |
| `scripts/otr*.py` | Order-to-Role entrypoints, currently on the [stimuli branch](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/stimuli/scripts) |
| `src/relational_lab/` | Shared package and instrumentation; OTR implementations live on its code branch |
| `investigations/` | Historical track-level overviews and indexes |
| `reports/` | Archived experiments, protocols, evidence and analysis summaries |
| `results/` | Local/gitignored outputs plus tracked historical outputs; PR #12 imported an earlier OTR results snapshot into `main`, with newer runs on the results branch |
| `docs/` | Shared reproducibility policies; OTR review figures currently on the stimuli branch |
| `tests/` | Root tests and branch-specific OTR tests |

## Current branches and integration boundaries

| Source | Intended treatment |
|:--|:--|
| `main` | Public navigation and established canonical material |
| [`experiments/exp102-clean-design`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/experiments/exp102-clean-design) | Review post-merge EXP-109–116 and PHASE2 additions independently |
| [Scratch study branches](learning-relations/EXPERIMENTS.md) | Fifteen current surface/diversity/recovery/mechanism/repair/origins studies plus position controls; pinned snapshots; no scientific branch merge in PR #11 |
| [`order-to-role/stimuli`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/stimuli) | Integrate OTR implementations, lightweight tests and documentation selectively |
| [`order-to-role/results-dev`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/results-dev) | Preserve development and test0 results for provenance; audit PR #12's existing snapshot and avoid bulk-importing more outputs |
| Other historical branches | Inspect whether they contain unique work before closing or archiving |

Use the [integration plan](../docs/INTEGRATION_PLAN.md) for sequencing and preservation checks.

## Active research boundary

ROLE-LEARNING-ORIGINS-v0 is ongoing locally. Preserve its active worktree, jobs, checkpoints and manifests. Paper I's evolving direction is distinct from Paper II; history is an archive, not a second home for current studies. The existing manuscript title and path are unchanged.

## Principles

1. Do not change frozen script bytes, paths, evaluation splits or protocol hashes merely to make folders look symmetrical.
2. Keep manuscript claims separate from exploratory observations, selected interventions and independent replications.
3. Record model revisions, generator versions, seed and split provenance, scoring rules and rights status when publishing results.
4. Keep branch-only files explicitly labeled until code and reports are actually integrated.
5. Do not automatically assign historical `ML-` studies to either paper based on a filename prefix.

**Documentation map, not a certification that all archived results are reproducible from `main`.**
