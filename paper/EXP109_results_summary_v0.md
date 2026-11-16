# EXP-109 — Prospective Replication Results

**Status:** Complete, within-grammar prospective replication.
**Design:** 8 independent seeds × 2 MPS numerical repetitions × 3 arms.
**Training:** Fixed 6,400-update endpoint; all runs retained.
**Evaluation:** 144 fresh AP anchors with prespecified transformations.

## Primary result: construction-sensitive failure

PA-only model, paired literal PA minus AP four-question accuracy:

- Mean difference: +0.6137 (+61.37 percentage points)
- Between-seed SD: 0.2397
- 95% t confidence interval: [0.4133, 0.8141]
- Positive effects: 8/8 independent seeds

This supports a substantial AP-versus-PA construction gap
within the synthetic grammar under the prespecified protocol.

## Secondary result: indirect exposure

Indirect PP+AA model minus PA-only model on unseen AP:

- Mean difference: +0.2296 (+22.96 percentage points)
- Between-seed SD: 0.2999
- 95% t confidence interval: [-0.0211, 0.4803]
- Positive effects: 6/8 independent seeds

The estimated benefit is positive but statistically inconclusive.

## Competence (descriptive, not used for filtering)

- PA-only: 6/16 runs
- Indirect PP+AA: 7/16 runs
- Direct PA+AP: 8/16 runs

## Interpretation and limitations

The primary result supports construction-sensitive relational
failure in this synthetic setting. It does not establish an
internal neural mechanism or broad language generalization.

The indirect intervention changes second-position passive
exposure relative to the PA-only control. It therefore does
not independently isolate abstract grammatical recombination.

MPS numerical variability was observed. Both repetitions
were included within each independent seed.

All 16 runs were retained without favorable checkpoint
selection. The original protected and within-event L+I
evaluations remain closed.

This document reports the original frozen analysis without
post-hoc modification of the experiment.
