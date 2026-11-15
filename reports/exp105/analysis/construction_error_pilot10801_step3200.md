# EXP-105 / construction-specific role-error diagnostic

**Status:** development-only, read-only checkpoint re-scoring of fixed step-3200 pilot seed 10801; no L+I, reserved, or protected scoring.

Reconstructed data and vocabulary match the original pilot hashes. All archived aggregate and per-voice metrics are reconciled with the saved pilot reports.

## Per-construction native four-name decoding

| Construction | Arm | n sentences | C_query | Per-question accuracy | Wrong event | Wrong role | Wrong role given correct event | Same-answer | Both correct / event |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | familiar | 216 | 25.46% | 61.46% | 20.14% | 18.40% | 23.04% | 19.91% | 42.82% |
| ALL | target | 216 | 31.48% | 73.84% | 3.70% | 22.45% | 23.32% | 34.95% | 53.70% |
| AA | familiar | 72 | 18.06% | 73.96% | 14.58% | 11.46% | 13.41% | 23.61% | 48.61% |
| AA | target | 72 | 47.22% | 82.64% | 3.12% | 14.24% | 14.70% | 20.14% | 69.44% |
| PA | familiar | 72 | 58.33% | 85.07% | 10.76% | 4.17% | 4.67% | 6.94% | 74.31% |
| PA | target | 72 | 29.17% | 75.35% | 4.51% | 20.14% | 21.09% | 35.42% | 54.86% |
| AP | familiar | 72 | 0.00% | 25.35% | 35.07% | 39.58% | 60.96% | 29.17% | 5.56% |
| AP | target | 72 | 18.06% | 63.54% | 3.47% | 32.99% | 34.17% | 49.31% | 36.81% |

Conditional wrong-role rate divides wrong-role questions by all questions selecting a member of the correct event. It is **not** the fraction of complete two-answer role reversals.

## Oracle-free joint-response taxonomy

The three disjoint categories below are calculated **only when both answers for an event name participants of that same correct event**. They are native model predictions *filtered by event membership*, not a re-decoding oracle that changes the candidate set.

| Construction | Arm | Events (2 × n) | Both choose correct event | Correct assignment given both | Reversal given both | Collapse given both |
|---|---|---:|---:|---:|---:|---:|
| ALL | familiar | 432 | 62.27% (269) | 68.77% | 2.23% | 29.00% |
| ALL | target | 432 | 92.82% (401) | 57.86% | 4.74% | 37.41% |
| AA | familiar | 144 | 71.53% (103) | 67.96% | 0.00% | 32.04% |
| AA | target | 144 | 93.75% (135) | 74.07% | 4.44% | 21.48% |
| PA | familiar | 144 | 79.86% (115) | 93.04% | 0.00% | 6.96% |
| PA | target | 144 | 91.67% (132) | 59.85% | 2.27% | 37.88% |
| AP | familiar | 144 | 35.42% (51) | 15.69% | 11.76% | 72.55% |
| AP | target | 144 | 93.06% (134) | 39.55% | 7.46% | 52.99% |

## Event-specific pair diagnostics

| Construction | Arm | Event | Both roles correct | Same-answer | Both answers from correct event | Reversal among those | Collapse among those |
|---|---|---|---:|---:|---:|---:|---:|
| AA | familiar | first | 27.78% | 36.11% | 63.89% | 0.00% | 56.52% |
| AA | familiar | second | 69.44% | 11.11% | 79.17% | 0.00% | 12.28% |
| AA | target | first | 52.78% | 29.17% | 88.89% | 7.81% | 32.81% |
| AA | target | second | 86.11% | 11.11% | 98.61% | 1.41% | 11.27% |
| PA | familiar | first | 79.17% | 2.78% | 81.94% | 0.00% | 3.39% |
| PA | familiar | second | 69.44% | 11.11% | 77.78% | 0.00% | 10.71% |
| PA | target | first | 48.61% | 41.67% | 94.44% | 4.41% | 44.12% |
| PA | target | second | 61.11% | 29.17% | 88.89% | 0.00% | 31.25% |
| AP | familiar | first | 11.11% | 33.33% | 41.67% | 3.33% | 70.00% |
| AP | familiar | second | 0.00% | 25.00% | 29.17% | 23.81% | 76.19% |
| AP | target | first | 40.28% | 41.67% | 93.06% | 11.94% | 44.78% |
| AP | target | second | 33.33% | 56.94% | 93.06% | 2.99% | 61.19% |

## Target minus familiar: percentage-point contrasts

| Construction | C_query | Question accuracy | Wrong event | Wrong role | Conditional wrong role | Same-answer |
|---|---:|---:|---:|---:|---:|---:|
| ALL | +6.02 | +12.38 | -16.44 | +4.05 | +0.27 | +15.05 |
| AA | +29.17 | +8.68 | -11.46 | +2.78 | +1.28 | -3.47 |
| PA | -29.17 | -9.72 | -6.25 | +15.97 | +16.42 | +28.47 |
| AP | +18.06 | +38.19 | -31.60 | -6.60 | -26.79 | +20.14 |

## Per-query accuracy (first agent, first patient, second agent, second patient)

| Construction | Arm | 1st agent | 1st patient | 2nd agent | 2nd patient |
|---|---|---:|---:|---:|---:|
| ALL | familiar | 40.74% | 85.65% | 64.81% | 54.63% |
| ALL | target | 75.46% | 62.50% | 89.35% | 68.06% |
| AA | familiar | 27.78% | 100.00% | 87.50% | 80.56% |
| AA | target | 79.17% | 66.67% | 94.44% | 90.28% |
| PA | familiar | 79.17% | 100.00% | 83.33% | 77.78% |
| PA | target | 83.33% | 59.72% | 83.33% | 75.00% |
| AP | familiar | 15.28% | 56.94% | 23.61% | 5.56% |
| AP | target | 63.89% | 61.11% | 90.28% | 38.89% |

## Interpretation boundaries

- One exploratory pilot seed; the denominators (72 sentences per construction) are not independent model-level replications.
- Construction-level means can reveal tradeoffs hidden by overall accuracy. No claim of a transferable model mechanism follows from these summaries.
- All per-construction outcomes use the original four-candidate decoding; unrestricted-vocabulary accuracy is separately recorded in JSON.
- No checkpoint selection, training, L+I rendering, reserved-case construction, or confirmatory-seed use.
- If a stored metric does not reproduce exactly, this audit raises an error rather than silently presenting an altered score.

