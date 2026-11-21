# POSITION-CB-v0 — complete raw-report audit and interpretation

**Source:** [frozen position-counterbalanced experiment](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/experiments/position-counterbalanced-roles-v0/reports/position_cb_v0) at branch commit `e6db0c23ea642cb773176f2dfb76d63d1f6da6ef`. **Status:** fully executed; eight independent seeds × two Apple MPS repetitions × two arms = 32 trained model states in 16 processes. No new scoring or model training was performed in this manuscript audit.

All 16 reports, the archived manifest and the exact `analysis_summary.json` were read using the connected GitHub source. I separately reconstructed the primary and gated-secondary estimates from the raw reports, checked source/evaluation SHA alignment and exhaustive first-/second-event pair categories, and verified that two MPS repeats were averaged **within** each seed. The independently reconstructed means and seed-level t intervals **match** the archived summary, including all digits reported below. All reports declare no protected/reserved/L+I use.

## Prespecified comparisons

| Test | Estimate | Two-sided seed-level Student t 95% CI | Seed sign | Frozen hierarchy |
|---|---:|---:|---:|---|
| Familiar minus unseen all-four-answer accuracy (`C_query`) averaged across arms | **+14.41 pp** | **[+0.0041, +28.82] pp** | 7/8 positive | Primary nominal pass, **near threshold** |
| Unseen minus familiar complete paired-role reversal | **+1.60 pp** | **[+0.0139, +3.19] pp** | 7/8 positive | Gated secondary nominal pass, **near threshold** |

Primary lower CI in percentage points: `0.004059121801425203`; secondary lower CI in percentage points: `0.013903352305924543`. Neither is a wide-margin positive result.

## Grouped descriptive results

| Training | Familiar voice combinations | C_query familiar | Unseen voice combinations | C_query unseen | Difference |
|---|---|---:|---|---:|---:|
| Same-voice (AA+PP) | AA, PP | 35.44% | AP, PA | 28.93% | +6.51 pp |
| Mixed-voice (AP+PA) | AP, PA | 38.54% | AA, PP | 16.23% | +22.31 pp |

The **interaction** of generalization gaps is **+15.80 pp**, exploratory paired seed-level CI **[-10.01, +41.61] pp**, **not established** because zero is within the interval and 4/8 individual seed differences are positive. The two arms' unseen voice sets differ, so comparing their raw generalization gap sizes alone is not a clean causal comparison of transfer difficulty.

## All independent seeds (means of the two numerical repeats)

| Seed | Same-voice familiar | Same-voice unseen | Mixed-voice familiar | Mixed-voice unseen | Across-arm gap |
|---|---:|---:|---:|---:|---:|
| 12701 | 5.03% | 1.91% | 98.26% | 24.48% | +38.45 pp |
| 12703 | 99.83% | 71.70% | 100.00% | 51.39% | +38.37 pp |
| 12707 | 1.91% | 2.43% | 0.87% | 0.87% | -0.26 pp |
| 12709 | 57.99% | 52.78% | 1.56% | 1.22% | +2.78 pp |
| 12713 | 8.33% | 8.85% | 100.00% | 44.97% | +27.26 pp |
| 12719 | 9.37% | 4.34% | 3.13% | 2.95% | +2.60 pp |
| 12721 | 1.22% | 0.69% | 2.95% | 2.43% | +0.52 pp |
| 12727 | 99.83% | 88.72% | 1.56% | 1.56% | +5.56 pp |

**Seed heterogeneity is severe:** around **90.3%** of the sum of the eight primary seed-level effects arises from seeds 12701, 12703 and 12713. Several independently trained models score near zero on even familiar constructions. This is not an exclusion criterion; **all eight must remain included**. The mean absolute within-seed difference between MPS repetitions on the primary contrast was **3.73 pp**, maximum **14.93 pp**. Descriptive leave-one-seed-out t intervals have a positive lower bound for only **2 of 8** possible omissions; this is an exploratory sensitivity check, not a different significance test.

## Full paired-error decomposition — exploratory

| Outcome | Familiar (pooled arms) | Unseen (pooled arms) | Unseen − familiar |
|---|---:|---:|---:|
| correct | 45.24% | 36.74% | -8.50 pp |
| reversed | 6.71% | 8.31% | +1.60 pp |
| collapsed | 10.21% | 12.56% | +2.35 pp |
| wrong event | 37.84% | 42.39% | +4.55 pp |

The 8.50-point decrease in both-roles-correct pairs is descriptively partitioned into a **+1.60 pp increase in reversals, +2.35 pp in same-person collapses, and +4.55 pp in wrong-event answers**. These are exhaustive *paired* categories, not independent hypothesis tests; the exact metric aggregation differs from four-query `C_query`.

Unseen individual-question accuracy falls by **6.30 pp**. Differences on error subtypes and arm interaction are exploratory, not multiple-comparison-adjusted confirmation.

## Threats and next decisions

- **Not a second grammar:** the same two-event grammar, vocabulary and familiar modifiers are reused; development examples have new source permutations in non-reserved name sets.
- **Matched in some respects, not all:** arms share the eight source cores/update, gold labels, active/passive voice marginals by event position and exact aggregate token bags; they intentionally differ in *within-source voice co-occurrence*, and not all absolute token-position distributions are matched.
- **Selection:** fixed 6400 updates, all seeds/repeats retained. Several models never became competent; reporting conditioned on competence would be post-hoc selection.
- **Statistics:** original threshold is a positive lower bound of a two-sided seed-level t 95% CI, with **seven degrees of freedom**. Borderline successes are statistically fragile under slight changes in training-seed sampling.
- **Mechanisms:** no unique role-binding circuit or generalization to new grammar families is demonstrated. Collapses/wrong-event increases support richer diagnostics, not a specific mechanistic explanation.

**Recommendation:** include this as a new main result and show the complete seed distribution in the paper. Do not launch another same-grammar head sweep. Before aiming at a main conference, independently replicate with greater robustness (for example, prospectively increase seed count or evaluate a truly separate syntax family under one frozen protocol). Any repeat is a new study, not a retrospective rescue of nominally significant outcomes.

**Source files:** [analysis_summary.json](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/experiments/position-counterbalanced-roles-v0/reports/position_cb_v0/analysis_summary.json), [manifest.json](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/experiments/position-counterbalanced-roles-v0/reports/position_cb_v0/manifest.json), 16 seed/repeat reports in the same folder.
