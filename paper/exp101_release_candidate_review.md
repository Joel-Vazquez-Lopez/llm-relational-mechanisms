# EXP-101 — R3 implementation release candidate, inspection only

**Status:** source-bound **pre-freeze** candidate. **Not authorized for any of the sixteen experimental training runs or protected evaluation.** This adds implementation validation to the existing R3 hypothesis; it does not revise the experimental contrast or expose new hypotheses.

## Complete implementation boundary

- `exp101_production_candidate.py` implements the fixed familiar-training/dev generation (62,208 train packets, 20,736 dev packets, 9 familiar cells), matched 64-question dose pools, ML-080-shaped 3 × 96 Transformer, and full 12,800-step AdamW routine. `train_one` is callable only after a separately reviewed release authorization, and **is not exposed through the CLI**.
- `exp101_runtime.py` is shared between the future study trainer and independent *artificial-tensor* optimizer/resume tests. Paired initial weights, base/dose sampling RNGs, optimizer tensors, checkpoint counters, and CPU RNG states are preserved. On CPU, two-step continuous versus one-step/checkpoint/resume/one-step agree exactly. This does **not** establish Apple MPS numerical determinism.
- `exp101_scoring.py` separates exact correctness from equivariance, scores nine familiar development cells, computes eight-seed paired bootstraps with 20,000 draws and synchronized seed-level comparisons, and explicitly distinguishes high performance/saturation from poor compositional transfer. The primary is **second-event C_voice on L+I**, not a claim about ordinal event-query binding.
- `exp101_protected_candidate.py` contains a **metadata-only future evaluation plan** (288 anchors; 23,040 future prompts/model, 368,640 total). Full rendering is internal and behind the approved source freeze, real checkpoint-based **joint** development gate, separate future protected unlock, and one-time pre-inference lock. It is **never called in this review**. Same-source four-query `C_query`, role-reversal, event-reordering, reference C_voice, and descriptive consistency scoring exist as pure functions.
- `exp101_release.py` creates/validates a single digest over actual source files (generator, independent validator, trainer, runtime, evaluation, scoring, release and tests), experiment constants, exact data-manifest hash, reference static report and future metadata-only test-plan digest. It refuses to authorize training if any source or data differs. The checked-in candidate status is deliberately `PREFREEZE_REVIEW_ONLY_NO_TRAIN_AUTHORIZATION`.

## Current hard blockers (do **not** override)

1. **Learned absolute position coverage:** all familiar training/development source+question sequences end at or before **34 tokens**. Development-only, unscored **L+I** renderings have lengths **37, 38 and 39**, with frequencies 6,144, 12,288 and 6,144 across 24,576 static packets. Therefore learned absolute embedding indices **34–38 (zero-based)** are never trained, even though the table has capacity 64. This could confound a negative generalization result. The approval guard refuses training until an independently approved, fixed length-coverage remedy is implemented, audited, and rehashed *before* training; no remedy has been silently selected here.
2. **Human grammatical acceptance:** independently review active/passive constructions, especially `Bero was helped near Dovan by Aven by Cira's gate` and the counterpart with a fronted gate adjunct. Verify that the two locatives remain semantically nonargumental; accept/reject templates and record a signed-off decision before freeze. Regex parser success is not a human grammatical judgment.
3. **Device-specific reproducibility and full integration:** synthetic CPU two-step exact resume and direct development-logit smoke testing pass, but actual 12,800-step study runs, step-800 crash recovery in the full generator, Apple MPS bitwise behavior and inference on real final checkpoints have **not** been executed. Freeze must select a supported device/reproducibility policy and inspect integration logs. No result claims are based on these unrun paths.

**A failed strong-success rule will not be labeled nontransfer.** A positive narrow primary effect does not itself warrant event-query binding claims; those require the independently reported `C_query` interpretation gate.

## Allowed verification without study training

```bash
python scripts/exp101_generator_preflight.py --phase preflight
python scripts/exp101_production_candidate.py --phase preflight
python scripts/exp101_protected_candidate.py --phase metadata-only
python scripts/exp101_release.py --phase check-candidate
python -m unittest discover -s scripts -p 'test_exp101_offline.py' -v
python -m unittest discover -s scripts -p 'test_exp101_release_offline.py' -v
```

No future freeze should be interpreted as approval until the source-bound manifest is reviewed and a **separate** explicit `frozen_release.json` is created after resolving the blockers. That approval file is deliberately absent.
