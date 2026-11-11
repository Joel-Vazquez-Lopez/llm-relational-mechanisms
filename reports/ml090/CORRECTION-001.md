# ML-090 — CORRECTION-001 (2026-11-11)

**Classification: implementation audit and retrospective correction; not a new confirmatory experiment.**

## Original records are immutable
- Frozen protocol: `reports/ml090/frozen_discovery_plan.json`
- Original output: `reports/ml090/discovery_evaluation.json`
- Original implementation: `scripts/ml090_role_cue_path.py`
- All must be preserved unchanged. No original published or reported numbers are overwritten.

## Reproduced defect
At original script lines 311–314, `logits[::2, ta[::2]]` selects a Cartesian set of source-row and target-token indices, yielding shape `[n_pairs, n_pairs]` under PyTorch advanced indexing. It is **not** a rowwise `[n_pairs]` paired contrast.

Independently tested under PyTorch 2.10.0 (CPU), using a 4 x 8 tensor: a 2 x 2 result versus a 2-element rowwise selection. An invariant-logits check can still pass with the bug; passing ML-090's role-cut invariant does **not** validate the statistic.

The intended measure for an agent/patient question pair `(2i, 2i+1)` is:
```
0.5 * [(z[2i,agent_i]-z[2i,patient_i])
     -(z[2i+1,agent_i]-z[2i+1,patient_i])]
```
Use explicit paired row indices. This statistic is in raw logit units, not normalized probabilities.

## Affected / unaffected
**Affected:** `mean_role_antisymmetric_evidence` in native and every intervention; `delta_role_evidence`; L2 who-minus-role comparisons; cross-exposure paired contrast, seed bootstrap and positive counts that use those values. Treat all original role-evidence numbers as **withdrawn pending replacement**.

**Not mathematically affected by this indexing expression:** `choice_accuracy`, `full_vocab_accuracy`, correct-vs-other margin, `mean_within_target_role_margin`, two-choice and wrong-event rates, native-error rescues, native-correct harms, `mean_output_logit_delta_l2`. These still require normal independent audit before any stronger claim. The calibration-only, previously examined model/sample status remains unchanged.

## Versioned reanalysis
- **Script:** `scripts/ml090_role_cue_path_v2_corrected.py`
- **New output ONLY:** `reports/ml090/correction_001_reanalysis.json` (created by an intentional run; not included with this document).
- Preflight: `python3 scripts/ml090_role_cue_path_v2_corrected.py --phase preflight`
- Reanalysis: `python3 scripts/ml090_role_cue_path_v2_corrected.py --phase run --device auto`
- The reanalysis loads only the original calibration cases and original ML-088 checkpoints. It verifies the original frozen plan and checkpoint protocols, then reproduces the original baseline/intervention metrics within 1e-4 before saving corrected numbers. Any parity failure **stops without creating a correction**.
- An existing correction output causes a refusal rather than overwrite.
- Commit the corrected output separately after inspection and retain file SHA-256 hashes.

## Scientific interpretation
This correction is retrospective reuse of opened models and calibration questions. It cannot support abstract role encoding or native causal necessity, even if a corrected role-dependent intervention effect survives. The proposed ML-097 study must be frozen, trained and independently evaluated as a separate prospective test.