# Final cross-cohort role-error audit (archived results only)

**Status:** completed *exploratory reanalysis* of previously opened protected evaluation **summaries** on 12 November 2026. **No new training, model inference, or protected-case generation.**

This report is the manuscript's final cross-cohort audit. Reproduction code: `scripts/paper_cross_cohort_audit.py`, which reads the five already archived JSON files and writes a provenance-logged JSON summary. The principal design and all original evaluation protocols predate this post-hoc comparison. This is not a new prospective test.

## Definitions and comparability

All five cohorts use the same synthetic two-event four-participant task family and four-name **native candidate scoring**, with eight matched training seeds and 768 protected target events per model arm. The reported *wrong-role rate* is the **question-level** fraction selecting the wrong participant of the **correct event**, not the fraction of **paired reversals**. The *wrong-event rate* is likewise per question. *Both correct* is the event-level fraction answering both agent and patient correctly. *Same person* is the event-level fraction with identical answers to the two complementary questions, regardless of which of the four participants was chosen.

Rates of wrong-role and wrong-event are mutually exclusive at the **question** level and sum with accuracy to 100%; *both correct* and *same person* describe **pairs** and are not an exhaustive alternative partition with question-level errors. These should never be stacked together as if they were three independent error outcomes.

ML-088's protected report lacks corresponding paired-response metrics, so it was **not** included. Its separately reported, previously examined **calibration-only oracle-restricted** reversal result is not directly pooled with these native four-choice measures.

## Results: paired target minus familiar exposure (percentage points)

| Cohort | Individual correct | Both-role correct | Wrong role per question | Wrong event per question | Same-person pairs |
|---|---:|---:|---:|---:|---:|
| ML-080 | +42.3 | +26.6 | -23.6 | -18.8 | +2.9 |
| ML-084 | +47.5 | +20.8 | -16.2 | -31.3 | +15.1 |
| ML-085 | +49.2 | +28.2 | -11.6 | -37.5 | +15.5 |
| ML-086 | +39.5 | +20.1 | -23.0 | -16.5 | +6.9 |
| ML-099 | +46.3 | +34.8 | -19.8 | -26.5 | +1.0 |
 
Across all **40 paired seeds**, individual and joint correctness each improve in **40/40**. Per-question wrong-role predictions decrease in **33/40**, per-question wrong-event predictions decrease in **34/40**, and native same-person pairs **increase in 26/40**. The mean same-person difference is positive in all five cohorts, despite its seed-level heterogeneity. Treat these as descriptive sign counts, **not** independent Bernoulli evidence across 40 distinct grammar families.

All five cohort-specific wrong-role differences are negative, yet individual seeds can still exhibit increases. The strong role-directionality contrast inferred from the **oracle-restricted ML-088 calibration audit** therefore does not turn into a universal claim that native paired-role reversals decreased under all cohorts: those precise joint-reversal counts were **not archived**.

## Interpretation

Targeted passive-second-event exposure consistently improves the ability to answer both complementary relational queries, even as same-person answers persist and, on average, increase under native four-choice scoring across these five cohorts. This is compatible with redistribution of answer errors and changing event-versus-role competition; it is **not** evidence of a uniquely identifiable role-binding computation. A greater native same-answer fraction can also arise when a model moves predictions from incorrect-event participants to the correct event without resolving agent-versus-patient selection. The aggregate reports do not permit a clean causal decomposition of these mechanisms.

There is a distinct, previously reported ML-072 three-arm study where candidate restriction was to **the correct event's two participants**, and targeted training nearly eliminated collapsed assignments. Since ML-072's grammatical/training design and candidate set differ, it cannot be pooled with this table. The behavioral claim should emphasize **task- and regime-dependent error topology**, not invariant error frequencies.

## Sources, provenance and reproducibility

These data are aggregated from exactly these immutable archived evaluation files in this repository (the reproducibility script computes plain SHA256 hashes of each file):

- `reports/ml080/protected_evaluation.json`
- `reports/ml084/protected_evaluation.json`
- `reports/ml085/protected_causal_evaluation.json`
- `reports/ml086/protected_evaluation.json`
- `reports/ml099_role_q/protected_evaluation.json`

Run from the repository root after obtaining the paper branch script:
```bash
python scripts/paper_cross_cohort_audit.py --phase preflight
python scripts/paper_cross_cohort_audit.py --phase audit
```

The script only reads the archived JSON evaluation summaries and writes `reports/paper_cross_cohort/archived_error_taxonomy.json`. It refuses to overwrite existing outputs. It does not import model code, generate cases, or load checkpoints.

**Stop rule:** this completes the cross-cohort check. No new independent-run claim, no further mechanic-screen exploration, and no protected data reuse beyond this declared retrospective analysis.
