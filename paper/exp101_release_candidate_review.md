# EXP-101 — RC2 pre-freeze implementation review

**Status:** inspection candidate only. No sixteen-run training launch, no study checkpoint evaluation, no protected item render/inference.

## Fixed changes since R3

1. Deterministic training-only absolute position-ID offsets 0..8 at update index t%9, identical across paired arms. Tokens/masks/labels unchanged; infer with offset=0. Actual familiar training prompts max 34, dev-only L+I max 39, hence evaluation indices 34..38 need explicit coverage. Static per-index nonpadding and *question-token* opportunities plus disposable gradient fixture are included. This modifies the training regime; no architecture sweep or model-outcome tuning.
2. Postposed gate locative replaced with **beside X's gate**, removing repeated adjacent `by` passives; fronted `By X's gate` retained. No claim that full discourse/pragmatic meaning remains identical. Independent human template review still pending.
3. CPU engineering fixture checks paired initial model weights and batch indices, 3-step uninterrupted vs checkpoint/resume equality including optimizer and schedule, checkpoint load → two familiar-dev event scores, and nonzero gradients at positions 34..38. Machine-specific Mac MPS fixture remains REQUIRED and cannot be completed in this Linux CPU container.
4. Candidate manifest binds actual implementation files, tests, protocol, grammar and position decision sheets, new static baseline report, split metadata, fixed configuration, and metadata-only protected count. The actual release authorization file is deliberately absent.

## Frozen scope unchanged

Only two arms, eight paired seeds, exactly 12,800 updates/model, nine familiar dev competence cells, one target-minus-familiar primary exposure contrast, bounded secondary cross-event correctness measure, same-cohort bare/L/I references, and one future protected unlock after the prespecified gate. No LLM evaluation or circuit intervention.

## Remaining independent blockers

**Independent human grammar review:** sign off 16 table configurations in `paper/exp101_grammar_review_RC2.md` and explicitly judge active/passive role preservation versus pragmatic meaning.

**Intended-hardware engineering fixture:** run `python scripts/exp101_hardware_fixture.py --device mps --output reports/exp101/mps_engineering_local.json` on the Mac in a temporary worktree with frozen source. Inspect PyTorch and macOS versions and numerical tolerances: parameter absolute 1e-5, relative 1e-4; first-step/restart loss absolute 1e-5; gradients all >1e-12. A fixture-only report does not itself authorize study training.

**Source-bound freeze decision:** after both independent gates, rebuild/check fingerprint and record an explicit human-reviewed approval file. Until then both training and protected evaluation refuse.
