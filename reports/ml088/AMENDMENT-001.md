# ML-088 Amendment 001 — Negative-control audit

Status: Post-calibration, pre-protected methodological audit.

The original shuffled-label control selected labels from each
example's four candidate participants. It therefore retained
input-dependent information and was not a valid no-information null.

Original control:
- Mean four-choice accuracy: 0.87888
- Mean 16-name accuracy: 0.8161

Corrected calibration-only audit:
- Globally permuted labels across training examples
- Training-label agreement: 0.06245
- Four-choice accuracy: 0.25285
- 16-name accuracy: 0.05718

The corrected control is consistent with chance.

The original frozen protocol, trained models, primary fitted
probes, logit-rehead baselines, and primary decision rule
remain unchanged.

This correction is an exploratory pre-protected amendment,
not an originally preregistered negative-control result.

No protected questions were accessed during this audit.

The main evaluation tests external linear recoverability,
not native routing necessity or causal mediation.
