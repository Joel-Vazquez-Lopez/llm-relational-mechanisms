[← Model Linguistics](../../README.md) · [Research history](../../investigations/README.md) · [Paper II](../order-to-role/README.md)

# Paper I · Relational Learning in Transformers

*Provisional navigation title.*

### Emerging direction: Learning to Bind — Seed-Dependent Emergence and Collapse of Relational Computation in Transformers

**Research question:** Why can Transformers with the same architecture, objective, data-generating distribution and training budget acquire different relational abilities?

**Model class:** Transformers trained from scratch on controlled synthetic relational tasks.

**Status (27 November 2026):** completed studies, exploratory follow-ups and an ongoing factorial learning-origins study. The original [working manuscript](../../paper/manuscript.md) has not been replaced.

## The scientific problem

Early controlled-exposure experiments repeatedly improved relational generalisation, while joint-role errors could persist. Subsequent studies varied construction position, surface complexity and relational diversity. Extended training revealed divergent trajectories: some models mastered the task; others learned which event was relevant but remained unable to select the correct participant within it.

The emerging question is how distinct relational abilities emerge, diverge and selectively fail to develop during learning.

## Start here

[**Results and figures**](RESULTS.md) · [Methods](METHODS.md) · [Mechanistic investigations](MECHANISMS.md) · [Scientific development](DEVELOPMENT.md) · [Integration audit](INTEGRATION.md)

## Research progression

| Stage | Question | Status |
|:--|:--|:--|
| A · Training conditions | How do surface complexity and role diversity affect acquisition? | Factorial results; primary diversity hypothesis not supported |
| B · Structured collapse | Can event competence emerge without role discrimination; can delayed models recover? | Selected recovery cohort and diagnostic follow-up |
| C · Computational organisation | Which representations and components support successful binding? | Exploratory localisation, causal routing, pathways and transfer |
| D · Independent replication | Do behavioural outcomes and candidate mechanism categories replicate? | Behavioural heterogeneity reproduced; original mechanism criterion failed; expanded search exploratory |
| E · Repair | Can targeted training restore role discrimination earlier or more specifically than controls? | Acceleration did not replicate; specificity inconclusive |
| F · Origins | Is divergence associated with initialisation, training stream or their interaction? | Ongoing 64-model factorial; no scientific results yet |

[All studies, source branches, protocols and implementations](EXPERIMENTS.md).

## Current evidence

In the independent replication, 9 of 16 models were successful, 3 intermediate and 4 fully collapsed at 51,200 updates. Collapsed models showed near-perfect event selection with approximately chance conditional role discrimination. These are outcomes in the studied task and cohort, not general prevalence estimates. The prespecified mechanistic classification failed; expanded discovery is exploratory. Targeted repair evidence remains limited and inconclusive.

These findings summarise the researcher-provided 27 November handover; see [evidence and limitations](EVIDENCE.md) for claim status and scope.

## Active study: ROLE-LEARNING-ORIGINS-v0

Eight initialisation seeds crossed with eight independent training-stream seeds, 64 models, each scheduled for 51,200 updates. Dense developmental measurements, event/conditional-role loss decomposition, gradient magnitudes and alignment, preserved model snapshots and resumable checkpoints investigate the origins and timing of divergent learning trajectories.

**Ongoing locally. Engineering pilot reported passed; no scientific findings yet.** See [experiment record](EXPERIMENTS.md#stage-f--current-learning-origins-study). Do not disturb the active research worktree or training jobs.

## Paper questions

1. Why do matched models diverge in relational competence?
2. Why can event selection develop without role binding?
3. When do successful and unsuccessful trajectories diverge?
4. Can early computational measurements predict collapse?
5. Can principled training interventions prevent it?
6. Does the phenomenon generalise beyond the studied synthetic grammar and architecture?

## Navigation

- [Evidence and limitations](EVIDENCE.md): positive, negative and inconclusive results.
- [Experiment index](EXPERIMENTS.md): foundations, position controls and all 15 handover studies.
- [Reproducibility](REPRODUCE.md): source snapshots, safeguards and active-run protection.
- [Existing manuscript](../../paper/manuscript.md): title and content unchanged.
- [Historical research](../../investigations/README.md): shared origins, not a duplicate of current studies.

Paper II investigates existing pretrained models; Paper I studies the acquisition of relational abilities during training. Similar functional dissociations motivate comparison but do not establish mechanistic equivalence.
