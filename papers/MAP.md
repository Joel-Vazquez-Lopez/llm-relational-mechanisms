[← Research guide](../STUDIES.md)

# Repository map and research ownership

The repository has **two paper-facing paths** under `papers/` and one shared technical codebase. This page records where each kind of file lives on `main` as of 28 November 2026. Canonical scripts, protected protocols and result paths are never moved for the sake of tidiness; this map explains the existing layout instead.

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

| Path | What it holds | Paper |
|:--|:--|:--|
| `papers/learning-relations/` | Paper I hub: reading guide, results, methods, mechanisms, evidence, experiment index, reproducibility, supplementary | I |
| `papers/order-to-role/` | Paper II hub: reading guide, evidence scorecards, experiments, layers, interventions, transfer, reproducibility; **curated** `results/`, `supplementary/` and checksummed `archive/` | II |
| `paper/` (singular) | Paper I working manuscript (`manuscript.md`) and historical ML/EXP/SLOG protocols. *Not the same as `papers/`.* | I |
| `docs/order_to_role/` | Paper II freeze and large-model protocols (FREEZE.md, FREEZE_XL.md, BIG_MODELS.md), input checksums, test0 attempt log, review package, state snapshots and figures. *Underscore name kept for path stability; the hub is `papers/order-to-role/`.* | II |
| `docs/` (other files) | Shared standards: evidence policy, provenance, workflow, research charter and log, experiment template, integration plan | Shared |
| `scripts/ml*.py`, `scripts/exp*.py` | Historical and Paper I experiment entrypoints. **A filename prefix does not decide paper ownership.** | I / history |
| `scripts/otr*.py` | Paper II entrypoints OTR-01 through OTR-15, plus battery, freeze scorer, generation checks and rescoring | II |
| `scripts/slurm/` | Cluster job scripts | Shared |
| `src/relational_lab/` | Shared package. `experiments/` contains Paper II stimulus generators and adapters (order-role, BX, XL, UD, WIKI-ROLES, Swedish) alongside shared instrumentation. | Shared |
| `tests/` | Unit tests, including the 13 OTR/order-role test files | Shared |
| `results/` | **Raw** run outputs (JSON, PNG, Slurm logs), flat, about 851 MiB of tracked files, mixing historical `ml*` and Paper II `otr*` runs. Cite curated tables in the paper hubs, not these files directly. Four Wikipedia-derived data files are deliberately not tracked. | Raw |
| `reports/` | Archived per-experiment reports, protocols and analysis summaries (mostly ML-series and Paper I) | I / history |
| `investigations/` | Historical chronology and the complete ML-series index | History |

## Branches and integration boundaries

| Source | Status | Treatment |
|:--|:--|:--|
| `main` | Canonical | Paper II code, protocols, tracked raw and curated results; Paper I manuscript and established studies |
| `order-to-role/stimuli` | **Merged into `main`** | Historical. Keep pinned links to `9c437928` for provenance; no new work here. |
| `order-to-role/results-dev` | Files imported into `main`; branch retained | **Keep:** pinned links to `25c6914e` depend on it. Read-only provenance; the four Wikipedia text files remain excluded from Git. |
| `experiments/exp102-clean-design` | Not merged | Paper I EXP-109–116 and PHASE2; review against its protocols before integrating |
| [Paper I study branches](learning-relations/EXPERIMENTS.md) | Mostly not merged | See the Paper I experiment index and [branch disposition](learning-relations/BRANCH_DISPOSITION.md) |
| Other branches | Mixed | Many are merged or superseded; see the integration plan. No branch deletion without archival review. |

Use the [integration plan](../docs/INTEGRATION_PLAN.md) for sequencing and preservation checks.

## Active research boundary

ROLE-LEARNING-ORIGINS-v0 is ongoing locally. Preserve its active worktree, jobs, checkpoints and manifests. Paper I's evolving direction is distinct from Paper II; history is an archive, not a second home for current studies. The existing manuscript title and path are unchanged.

## Principles

1. Do not change frozen script bytes, paths, evaluation splits or protocol hashes merely to make folders look symmetrical.
2. Keep manuscript claims separate from exploratory observations, selected interventions and independent replications.
3. Record model revisions, generator versions, seed and split provenance, scoring rules and rights status when publishing results.
4. Label any file that exists only on a branch. Once it is merged, link to the `main` path and keep a pinned-commit link only where provenance needs it.
5. Do not automatically assign historical `ML-` studies to either paper based on a filename prefix.

**Documentation map, not a certification that all archived results are reproducible from `main`.**
