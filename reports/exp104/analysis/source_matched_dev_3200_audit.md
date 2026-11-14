# EXP-104 — source-matched scheduling, complete development audit

**Status:** exploratory; frozen 3,200-step endpoint, eight paired seeds. No protected, LI, or reserved examples generated.

Protocol SHA256 (canonical JSON): `2ae8d40fe2ae6d7f943f5e0ed6cc6f2e85484c5bf0f2d0ebba55b542f0fe8825`

## Primary endpoint: DEV C_query at step 3200

| Seed | Quartet | Distributed | Paired difference |
|---:|---:|---:|---:|
| 10601 | 95.14% | 56.25% | +38.89 pp |
| 10603 | 86.81% | 82.29% | +4.51 pp |
| 10609 | 76.74% | 100.00% | -23.26 pp |
| 10613 | 94.10% | 99.31% | -5.21 pp |
| 10621 | 99.31% | 100.00% | -0.69 pp |
| 10627 | 76.04% | 99.65% | -23.61 pp |
| 10629 | 81.94% | 32.99% | +48.96 pp |
| 10633 | 93.40% | 98.61% | -5.21 pp |

Quartet mean: **87.93%**; distributed mean: **83.64%**.
Mean paired difference: **+4.30 pp** (3,200-step endpoint).
Descriptive paired-seed bootstrap 95% interval: **[-11.89 pp, +22.35 pp]**.
Wins/ties/reversals: **3/0/5**. Two-sided exact sign-flip p (exploratory only): 0.6719.

## Learning trajectories (secondary)

| Step | Quartet mean C_query | Distributed mean C_query | Paired gap | Positive pairs |
|---:|---:|---:|---:|---:|
| 400 | 0.13% | 0.00% | +0.13 pp | 1/8 |
| 800 | 14.63% | 0.13% | +14.50 pp | 7/8 |
| 1200 | 30.86% | 2.95% | +27.91 pp | 8/8 |
| 1600 | 56.08% | 40.89% | +15.19 pp | 6/8 |
| 2000 | 71.79% | 51.30% | +20.49 pp | 5/8 |
| 2400 | 76.43% | 62.07% | +14.37 pp | 4/8 |
| 2800 | 86.11% | 76.74% | +9.38 pp | 2/8 |
| 3200 | 87.93% | 83.64% | +4.30 pp | 3/8 |

Average over eight sampled development checkpoints (secondary, not time-integrated AUC): +13.28 pp; descriptive bootstrap 95% interval [-2.69 pp, +29.08 pp]; 6/8 seed-pairs positive.

## Secondary errors and accuracy at step 3200

| Metric | Quartet mean | Distributed mean | Quartet − distributed |
|---|---:|---:|---:|
| accuracy4 | 96.51% | 94.56% | +1.94 pp |
| same_answer_rate | 2.76% | 6.66% | -3.91 pp |
| wrong_event_rate | 1.45% | 1.14% | +0.31 pp |
| wrong_role_rate | 2.04% | 4.30% | -2.26 pp |
| per_event_both_correct | 93.73% | 90.30% | +3.43 pp |

## Predeclared familiar-development gate

Quartet: **4/8**; distributed: **5/8**. Overall: **FAIL** (requires ≥6/8 in both).

## Interpretation

- The original primary endpoint shows no consistent across-seed quartet advantage under source–question matching.
- Earlier milestones suggest a possible learning-speed advantage; this is secondary and exploratory, not evidence of a guaranteed benefit.
- Different order, per-update source diversity, gradient covariance, and optimization trajectories remain inseparable in this intervention.
- Source–question multisets are matched within four-update windows, not within individual gradient updates.
- This study cannot establish equivalence or explain the original EXP-102 gap uniquely. The failure of the competence gate does not authorize test access.
- Neither L+I nor reserved data were used; all comparisons remain familiar-development only.

**PASS:** 16/16 completed reports; fixed seed pairs and eight milestones; hashes, metric identities and protected flags verified.
