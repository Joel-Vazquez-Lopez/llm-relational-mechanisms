# EXP-101 MPS RC3 resume-failure diagnostic (engineering only)

**Status:** unreleased diagnostic branch, EXP-101 study remains unauthorized. No study seeds, full training, or protected examples are used.

## Failure observed on intended Mac

- RC3 release hash and positional coverage checks passed.
- The RC3 MPS fixture passed its first paired-loss and sampled-index checks, then failed its post-three-step assertion: `Resume mismatch after 3 fixture steps`.
- **Do not interpret that assertion as proof of faulty checkpoint restoration.** RC3 runs two independent first updates, from matched initial weights, and does not compare resulting model/optimizer states until after resuming one of them.
- In MPS `warn_only` mode, nondeterministic backward kernels are permitted; independently executed updates can produce diverging floating-point states even with the same sampled indices.

## Added root-cause diagnostic

Run `scripts/exp101_mps_resume_diagnostic.py` with `--device mps` on the intended Mac.

It records:
1. Exactly identical initial model, optimizer, and batch RNG states.
2. Post-first-step model **and optimizer** differences between independent training trajectories (`a`, `b`), with existing RC3 `rtol=1e-4, atol=1e-5`; no threshold change.
3. Exact model/optimizer/batch-RNG equality immediately after saving `b` and restoring into `c`.
4. Step-by-step differences after continuing uninterrupted `a` vs restored `c`.
5. Differences between two separately restored continuations from the **same saved `b` state**.
6. Warnings, environment metadata, numerical magnitudes, and pass flags; the report is written even when tolerances fail. It does not classify a diagnostic run as release approval.

This distinguishes initial MPS computation divergence from checkpoint round-trip corruption, and tests whether independent MPS continuations are numerically stable within the *existing* tolerances.

## CPU control executed

On an available CPU backend (`torch 2.10.0+cpu`), the same diagnostic completed. Model differences after the independent first step, immediate save/load, uninterrupted vs resumed continuation, and two independent resume branches were each **0.0**, and optimizer states agreed. This is only a CPU fixture control; it cannot validate Apple MPS.

## Required interpretation of Mac result

- **First-step models differ, save/load is exact:** initial MPS nondeterminism is a confound; original RC3 assertion cannot isolate a resume bug.
- **Save/load differs immediately:** investigate serialization, MPS device migration, optimizer tensor states, and RNG handling before any release.
- **Two branches restored from the same checkpoint differ:** MPS computational nondeterminism persists even after exact restoration. Quantify magnitude; do not claim bitwise replay.
- **Within-tolerance model states but optimizer mismatches:** inspect individual AdamW states; do not waive the optimizer requirement.
- **All comparisons within tolerance:** record two independent fresh-process runs, platform/PyTorch details and warnings for review. It still does not prove long-run reproducibility.

The study remains locked, human grammar review remains outstanding, and there is **no modification to the RC3 release fingerprint, training code, confidence thresholds, or protected-evaluation access** on this diagnostic branch.
