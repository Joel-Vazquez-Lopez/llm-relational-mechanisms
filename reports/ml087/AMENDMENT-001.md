# ML-087 Amendment 001 — Duplicate initialization

## Status
Orchestration-only bug fix, recorded before evaluation.

## Original failure
The original `main()` calls `prepare()` and then `run()`
calls `prepare()` a second time.

The second call fails the vocabulary-disjointness assertion
inside ML-086's setup function.

## Scientific impact
The failure occurred before loading or evaluating checkpoints.

No protected cases were generated.

No intervention results were observed before this amendment.

## Correction
`scripts/ml087_evaluate_amendment.py` calls `prepare()`
once and reuses its result when invoking the original `run()`.

The frozen ML-087 implementation, evaluation questions,
model checkpoints, subspace extraction, interventions,
control distributions, primary statistic, and success
criterion are unchanged.

The original frozen script and protocol remain unmodified.

## Classification
Implementation amendment, not hypothesis revision.

The resulting evaluation remains exploratory and
calibration-only, using the existing ML-086 models.
