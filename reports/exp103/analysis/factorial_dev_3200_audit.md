# EXP-103: Source-matched factorial study — development-only audit

**Status:** descriptive, exploratory, based on frozen step-3200 endpoints. No protected or reserved cases generated.

## Final AP (first active, second passive): C_query

| Exposure / update schedule | Mean across 8 seeds | Seeds ≥ 90% |
|---|---:|---:|
| familiar / quartet | 8.68% | 0/8 |
| familiar / distributed | 8.85% | 0/8 |
| target / quartet | 33.33% | 0/8 |
| target / distributed | 33.16% | 0/8 |

## Paired factorial contrasts on AP C_query

| Contrast | Mean effect | Descriptive paired-seed bootstrap 95% CI | Positive seeds |
|---|---:|---:|---:|
| exposure | +24.48 pp | [+13.45 pp, +35.94 pp] | 8/8 |
| policy | -0.00 pp | [-11.02 pp, +9.46 pp] | 5/8 |
| interaction | +0.35 pp | [-12.67 pp, +12.85 pp] | 4/8 |

**Definitions:** exposure = mean(target) − mean(familiar) across policies; policy = mean(quartet) − mean(distributed) across exposure arms; interaction = (target quartet − target distributed) − (familiar quartet − familiar distributed).

**Per-seed paired primary endpoint:**

| Seed | Exposure effect | Policy effect | Interaction |
|---|---:|---:|---:|
| 10501 | +4.17 pp | +1.39 pp | -2.78 pp |
| 10503 | +5.56 pp | +5.56 pp | +11.11 pp |
| 10509 | +15.97 pp | +17.36 pp | -20.83 pp |
| 10513 | +42.36 pp | +7.64 pp | +4.17 pp |
| 10521 | +12.50 pp | -30.56 pp | +13.89 pp |
| 10527 | +26.39 pp | +15.28 pp | +30.56 pp |
| 10529 | +43.75 pp | -15.97 pp | -31.94 pp |
| 10533 | +45.14 pp | -0.69 pp | -1.39 pp |

## AP error taxonomy at step 3200

| Condition | Question accuracy | Wrong event | Wrong role | Same-answer | Two-query both-correct |
|---|---:|---:|---:|---:|---:|
| familiar/quartet | 48.78% | 16.58% | 34.64% | 20.05% | 32.55% |
| familiar/distributed | 49.61% | 18.27% | 32.12% | 24.05% | 30.99% |
| target/quartet | 71.53% | 5.95% | 22.53% | 18.58% | 59.11% |
| target/distributed | 69.14% | 7.81% | 23.05% | 16.75% | 57.29% |

## Secondary paired contrast summaries

| AP outcome | Exposure main effect | Policy main effect | 2×2 interaction |
|---|---:|---:|---:|
| accuracy4 | +21.14 pp | +0.78 pp | +3.21 pp |
| wrong_event_rate | -10.55 pp | -1.78 pp | -0.17 pp |
| wrong_role_rate | -10.59 pp | +1.00 pp | -3.04 pp |
| same_answer_rate | -4.38 pp | -1.09 pp | +5.82 pp |
| per_event_both_correct | +26.43 pp | +1.69 pp | +0.26 pp |

## Familiar development competence gate: AA and PA

Individual model passes only when **both AA and PA** have question accuracy ≥95% and per-event both-role correctness ≥90%. Predeclared requirement: ≥6 of 8 seeds in **every arm**.

| Arm | Passing seeds | Meets ≥6/8 |
|---|---:|---:|
| familiar/quartet | 6/8 | YES |
| familiar/distributed | 8/8 | YES |
| target/quartet | 6/8 | YES |
| target/distributed | 6/8 | YES |

**Overall gate:** PASS. A passing gate is not permission to open reserved evaluation. This runner has no protected-evaluation phase.

## AP learning trajectories (mean C_query)

| Step | Familiar/Q | Familiar/D | Target/Q | Target/D |
|---:|---:|---:|---:|---:|
| 800 | 1.74% | 1.04% | 4.17% | 2.95% |
| 1600 | 6.60% | 5.21% | 20.66% | 18.40% |
| 2400 | 7.29% | 9.03% | 27.08% | 23.96% |
| 3200 | 8.68% | 8.85% | 33.33% | 33.16% |

## Interpretation and limitations

- The unit of replication is the paired random seed (n=8), not the 72 examples per dev cell.
- These are **development** comparisons after exploratory study design; intervals are descriptive, not preregistered confirmation.
- The sources/questions are exactly matched **within each four-update window** across schedule policies, but not within individual updates or optimizer trajectories.
- Target versus familiar exposure intentionally changes the training construction. This is not a direct ML-080 replication.
- Do not infer that absence of mean scheduling effect proves equivalence. A zero-looking mean may hide seed-level heterogeneity.
- The AP set uses familiar modifiers and held-out name sets; it is not the protected L+I composition evaluation.
- The historical outcomes and EXP-102 results should remain immutable; no post hoc checkpoint selection or seed exclusion.

PASS: 32/32 frozen endpoint reports, 8 paired seeds, 4 arms, 3 dev cells, 4 time points, metric identities and protected-access assertions.
