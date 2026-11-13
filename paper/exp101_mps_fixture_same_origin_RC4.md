# EXP-101 RC4: same-origin MPS fixture correction (review only)

**Scope:** This changes only the engineering fixture and its documentation/tests.
Scientific hypothesis, grammatical templates, exposure contrast, training path,
model architecture, data, position offsets, scoring, confidence intervals,
seed allocation, checkpoint policy, numerical tolerances and stop rules are unchanged.
**It does not authorize study training or protected evaluation.**

## Why the RC3 fixture failed

The RC3 fixture initialized models A/B and executed a separate one-step
update for each, saved B's checkpoint, then compared resumed B against
uninterrupted A at step 3. This confounded cross-run MPS numerical variation
with checkpoint restoration. The MPS check failed its RC3 comparison.

A subsequent *user-reported* same-origin engineering diagnostic on the intended
Mac (commit `3665871e95f28a7471380bda136bca9f1a398e86`) found:
- Save/load model and optimizer state: exact.
- Same-origin uninterrupted versus resumed parameter max |difference|:
  2.9802322387695312e-08; within the existing tolerances.
- Two independent resumes from the same checkpoint: 2.9802322387695312e-08;
  within tolerance.
- 30 nondeterministic-kernel warnings under the MPS warning-only policy.

These claims come from the reported terminal summary; the full local JSON has
**not** been independently inspected or archived here.

## Fixed fixture

1. Initialize two runs identically and evaluate *independent first-step* parity
   separately, recording its max difference and optimizer agreement.
2. Save **A's own** step-one checkpoint; restore its model/optimizer/sampler RNG
   into C. Require exact step-one model, optimizer and sampler round-trip.
3. Continue A without restoring and C after restoring through the same fixed
   two updates (offsets 1 and 2). Require identical sampled batches and
   |loss delta|<=1e-5; compare all model tensors with
   `torch.allclose(rtol=1e-4, atol=1e-5)` and all optimizer fields with
   the same tolerances.
4. Retain the familiar-development checkpoint-to-metric smoke and gradient
   coverage on indices 34–38, along with explicit MPS warning reporting.
5. Do not treat the diagnostic as proof of long-horizon determinism.

## Release gates still pending

Run the RC4 fixture twice in independent processes on the **intended Mac MPS**;
archive both full JSON files, stdout/stderr, torch/python versions, numerical
comparisons, and warnings. A reviewer must explicitly approve the documented
MPS warning-only policy and tolerances. Independent human grammar sign-off
and a separately reviewed frozen release manifest remain mandatory. No study
seed training or protected access is authorized.
