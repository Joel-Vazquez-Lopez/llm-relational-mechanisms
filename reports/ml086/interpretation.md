# ML-086 — Post-hoc representation audit

**Status:** Exploratory analysis performed after the
ML-086 protected evaluation. Not a predeclared primary test.

## Frozen experiment

- Protocol hash: `f90585d299bd5f5b2a9bd3fb3f94bde8e026e370cbac7501b8eba5818586612d`
- Intervention site: layer 2 attention, final answer token
- Subspace rank: 8
- Eight paired seeds, two training conditions
- Original protected results remain unchanged

## Original prospective results

- Mean behavioral exposure benefit:
  0.3947
- Positive behavioral pairs:
  8/8
- Mean primary mechanistic contrast:
  0.0527
- Positive mechanistic comparisons:
  6/8
- Bootstrap 95% interval:
  [-0.0150, 0.1305]
- Frozen primary success rule:
  True

The mechanistic rule was met narrowly, but its bootstrap
interval includes zero. This is suggestive rather than
conclusive evidence of selective dependence.

## Calibration-only geometry

| Measurement | Familiar | Target |
|---|---:|---:|
| Raw role/event overlap | 0.8394 | 0.7398 |
| Main-effect overlap | 0.6898 | 0.6327 |
| Interaction/role-basis overlap | 0.9270 | 0.8273 |
| Interaction/event-basis overlap | 0.8658 | 0.8429 |
| Role-difference interaction energy | 0.4359 | 0.3236 |
| Event-difference interaction energy | 0.4790 | 0.4193 |

Expected overlap of independent random 8D subspaces
in 96D: 0.0833.

## Interpretation

1. Role and event subspaces overlap strongly.
2. Separating factorial main effects reduces the measured
   overlap, but substantial alignment remains.
3. The factorial interaction contributes considerable
   activation-difference energy in many models.
4. The pattern is heterogeneous; it is not a universal
   large-interaction mechanism.
5. Role- and event-subspace erasures show similar
   behavioral effects, including on familiar grammar.
6. The intervention is therefore not yet uniquely
   attributable to role binding, nor is a specific
   mediation pathway identified.

The energy fractions describe squared activation
differences, not explained behavioral variance.

The factorial interaction also need not uniquely encode
binding: nonlinear responses to question cues can
produce the same mathematical structure.

## Reproducibility

Run:

    python3 scripts/ml086_posthoc_audit.py

Uses only frozen familiar calibration data and existing
checkpoints to recompute geometry.

The existing protected JSON is read for aggregation only;
no protected questions or model inference are generated.

Machine-readable results:
`reports/ml086/posthoc_audit.json`

Frozen prospective results:
`reports/ml086/protected_evaluation.json`

## Next hypothesis

A future independent cohort should compare the causal
effects of interaction-associated directions with
main-effect and matched-random directions.

Such an experiment must be frozen independently before
evaluating unfamiliar grammar.
