# EXP-101 RC3: Mac MPS determinism limitation and proposed policy amendment

**Status: review-only. This does not authorize training or protected evaluation.**

## Observed problem

On the intended Mac, the RC2 engineering fixture passed the source-bound release check but failed during its first `loss.backward()` because `index_put_with_accumulate_mps` has no deterministic MPS implementation when PyTorch strict deterministic mode is on. This was a disposable hardware check, not EXP-101 model training.

## Single fixed remedy to test

- CPU uses `torch.use_deterministic_algorithms(True, warn_only=False)`.
- Mac MPS uses `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Both the disposable hardware fixture and the locked study-training path call `exp101_runtime.configure_determinism(device)`; they must never diverge silently.
- MPS warning-only is **not deterministic**; it merely allows unsupported deterministic operations to run with a warning. This changes the reproducibility assumption but does not change architecture, grammar, split, model initialization seeds, exposure contrast, positional schedule, loss, optimizer or task metrics.
- Keep prespecified hardware tolerances: parameter rtol=1e-4, atol=1e-5, loss difference at most 1e-5. Check both AdamW optimizer state and model parameters after an interrupted/resumed execution. Report actual maximum parameter difference and PyTorch nondeterministic warnings, not just a binary pass.
- Do not relax tolerances after examining a failed fixture without another separately reviewed amendment.

## Before any study run

Run the same three-update engineering fixture on the intended MPS Mac, preserve stdout/stderr and JSON, and rerun from another fresh process. A pass establishes only **short-run numerical agreement to tolerance**, not exact deterministic long-run reproducibility. Independent review must accept or reject this execution policy. A fully strict CPU training alternative would require explicit separate hardware/budget approval, not a silent fallback.

Independent grammatical sign-off and final source-bound release approval are still required. No study seeds, protected examples or experimental runs may be started under RC3.
