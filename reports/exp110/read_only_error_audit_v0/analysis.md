# EXP-110 — Read-only error audit of EXP-109

**Status:** post-hoc descriptive analysis of all 16 frozen EXP-109 reports. Not a fresh confirmatory test. No training, checkpoint reading, LI, reserved or protected evaluation.

## Frozen endpoints (unchanged)

| Effect | Seed-level mean | 95% t interval (8 seeds) |
|---|---:|---:|
| PA-only: literal PA − AP | +61.4 pp | [+41.3 pp, +81.4 pp] |
| Indirect AP − baseline AP | +23.0 pp | [-2.1 pp, +48.0 pp] |

## Native scores by construction (averages of seed means)

| Arm / construction | All four correct | Question accuracy | Wrong event | Wrong role | Same answer |
|---|---:|---:|---:|---:|---:|
| pa_only / AP | 6.3% | 47.3% | 14.0% | 38.7% | 19.9% |
| pa_only / PA_literal | 67.7% | 89.0% | 4.7% | 6.3% | 9.0% |
| indirect_pp_aa / AP | 29.3% | 66.6% | 8.5% | 24.8% | 20.9% |
| indirect_pp_aa / PA_literal | 61.1% | 84.6% | 3.8% | 11.5% | 13.1% |
| direct_pa_ap / AP | 35.8% | 71.8% | 8.6% | 19.6% | 21.7% |
| direct_pa_ap / PA_literal | 64.0% | 86.7% | 4.3% | 9.0% | 11.9% |

## Baseline AP − literal PA: paired descriptive error changes

| Metric | Seed-mean difference | 95% t interval |
|---|---:|---:|
| accuracy4 | -41.7 pp | [-50.4 pp, -33.0 pp] |
| full_vocab_accuracy | -42.0 pp | [-50.7 pp, -33.3 pp] |
| wrong_event_rate | +9.2 pp | [+2.5 pp, +16.0 pp] |
| wrong_role_rate | +32.4 pp | [+25.8 pp, +39.1 pp] |
| same_answer_rate | +10.9 pp | [+6.8 pp, +14.9 pp] |

## AP second-event agent/patient pair taxonomy

Mutually exclusive outcomes under native four-name candidate predictions, averaged across seed-level proportions. These are **not** oracle-restricted outcomes.

| Arm | Both correct | Reversed | Collapsed | Wrong event name |
|---|---:|---:|---:|---:|
| pa_only | 11.1% | 41.2% | 21.4% | 26.4% |
| indirect_pp_aa | 36.4% | 17.1% | 27.2% | 19.3% |
| direct_pa_ap | 43.8% | 10.8% | 27.8% | 17.6% |

## AP joint outcomes after literal clause swap

| Arm | AP wrong, PA correct | AP correct, PA wrong | Both correct | Both wrong |
|---|---:|---:|---:|---:|
| pa_only | 62.8% | 1.5% | 4.9% | 30.8% |
| indirect_pp_aa | 36.2% | 4.5% | 24.8% | 34.5% |
| direct_pa_ap | 33.6% | 5.5% | 30.3% | 30.6% |

## MPS numerical variability (repeat 2 − repeat 1)

These are absolute differences in outcome proportions, not independent seed effects.

| Outcome | Mean absolute difference | Maximum difference |
|---|---:|---:|
| primary_pa_only_gap | 13.7% | 44.4% |
| secondary_indirect_AP_lift | 14.8% | 41.7% |

## Interpretation and boundaries

- The original PA-only AP/PA gap is reproduced, and the AP error excess is dominated numerically by wrong-role answers rather than wrong-event answers.
- AP second-event pair reversals and collapses are distinguishable behavioral outcomes; this does not identify an internal role-binding circuit.
- Comparisons in this document beyond the two frozen endpoints are **exploratory**. Their confidence intervals are descriptive and not adjusted for multiple comparisons.
- No successful checkpoints were selected, and no competence filtering was applied.
- Second-position voice, sentence-position heuristics and optimization remain alternative explanations.
- The original reserved and L+I tests remain closed.

## Falsifiable mechanistic *hypothesis* (not a result)

A model that has learned a first-event-passive role heuristic may fail to invert agent/patient mapping when passive syntax moves into the second event. Before circuit claims, compare role-specific logit margins and controlled within-model interventions on matched clause-swap counterfactuals, with position/event/perturbation controls and independently frozen replication.

## Provenance

- Frozen EXP-109 commit: `151e0035b6451dcbf65f264f7a0e14ebc0b981d3`
- Protocol digest: `bd5eafe3bc1958878c63e4ce41c8bc1e29b1585b8458814f46ac516f18aa209f`
- Evaluation digest: `5063bec0fceca42ce9a899ffeaa93c0e26a56b93734e80d5388e211e7c9e0b76`
- Sources: all 16 JSON reports in `reports/exp109/replication_v0/`
- This analyzer reads source reports only; it does not generate or score examples.
