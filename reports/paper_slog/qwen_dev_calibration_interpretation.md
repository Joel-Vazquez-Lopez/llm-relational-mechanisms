# SLOG/Qwen development-only calibration — measurement suitability result

**Date:** 2026-11-12. **Status:** completed public **in-distribution development** calibration; **not** a structural-generalization benchmark result. No protected SLOG generalization archive or test examples were opened.

## Model and execution

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Exact model commit: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- PyTorch: `2.15.0`; Transformers: `5.20.2`; `device=mps`; Apple Silicon Mac with 16 GiB unified RAM
- Audit source: `scripts/paper_slog_dev_model_calibration.py`, script SHA256 `44a1ce8a937645aa43fcb2d15f17a3fb3fec9b4ca546b8c3d2f5cbb30a312e5b`
- Dataset: official SLOG `data/cogs_LF/dev.tsv`, SHA256 `177128c3f64be56732a102db119139bbec33a771c485a525ea0f9f4e45926a5c`
- Deterministic sample: 16 development sentences from 149 eligible `in_distribution` multi-predicate examples. Manifest SHA256 `7c3e953872620451c5e43c2534149c5b9cf71555c78188343ba4d16cc05f64b1`.
- 64 question variants = two role questions (agent/theme) × two option orders × 16 sentences. Token IDs: `A=32`, `B=33`. Method: compare final-position chat-template next-token logits of `A` versus `B`, without free-text generation. Choices are the annotated agent/theme heads only, with the event verb marked.
- Local detailed per-example report on the user's Mac: `reports/paper_slog/dev_model_calibration.json`. This file includes source sentences/gold labels, is intentionally gitignored, and has **not** been uploaded to GitHub.

## Reported outcomes

| Measure | Result | Units/interpretation |
|---|---:|---|
| Individual questions correct | 43/64 (67.19%) | Across all 64 prompt/order variants; not 64 independent sentences |
| Both agent and theme correct under first option order | 4/16 (25.00%) | Event-level, option order 0 only |
| Same candidate for both roles under first order | 12/16 (75.00%) | Event-level, option order 0 only |
| Both roles reversed under first order | 0/16 (0.00%) | Event-level, option order 0 only |
| Both role answers invariant across A/B order | 3/16 (18.75%) | **Low measurement robustness** |
| First-order both-correct and order-invariant | 0/16 | All four initially both-correct examples changed at least one selected participant with A/B order reversal |

Per-sentence first-order paired outcome and whether BOTH role responses are option-order invariant:

| Dev row (1-based) | Both correct | Same person | Reversed | Both roles invariant to option order |
|---:|---:|---:|---:|---:|
| 3308 | 0 | 1 | 0 | 1 |
| 1101 | 0 | 1 | 0 | 0 |
| 2177 | 1 | 0 | 0 | 0 |
| 824 | 0 | 1 | 0 | 0 |
| 610 | 0 | 1 | 0 | 1 |
| 1496 | 0 | 1 | 0 | 0 |
| 863 | 0 | 1 | 0 | 0 |
| 3262 | 0 | 1 | 0 | 0 |
| 188 | 1 | 0 | 0 | 0 |
| 2404 | 0 | 1 | 0 | 0 |
| 3223 | 1 | 0 | 0 | 0 |
| 2605 | 0 | 1 | 0 | 1 |
| 1756 | 0 | 1 | 0 | 0 |
| 430 | 1 | 0 | 0 | 0 |
| 628 | 0 | 1 | 0 | 0 |
| 4007 | 0 | 1 | 0 | 0 |

## Interpretation and stopping decision

The calibration **ran successfully**, the frozen model revision was resolved, A and B each tokenized as a single token, and the Apple Silicon MPS execution worked. However, the **measurement method is not sufficiently robust for protected evaluation**: for 13/16 sentences, reversing A/B candidate order changes at least one of the two role assignments. The provisional aggregate `order0_collapse=0.75` is therefore **not evidence** of representational role collapse in pretrained Qwen. The observed 67.2% marginal choice accuracy is an exploratory sanity-check on a very small, highly correlated public-development sample and cannot support claims of OOD generalization, successful role binding, or lack thereof. This was **not a prespecified hypothesis test**, and no formal preregistered competence threshold was established.

Possible causes include A/B choice-position or token-format sensitivity, instruction-following details, and incomplete/ambiguous mapping of sentence LF roles to single-word surface heads. These hypotheses are **not distinguished** by the available logged aggregates; the output should not be interpreted as proof of a specific option bias or of underlying semantic incompetence.

**Stop:** no Qwen retuning, further pretrained-model experiments, or structural-generalization archive access for the current manuscript. An independent future study would require a prevalidated option-order-invariant scoring rule, more thorough annotation projection audit, frozen prompts/model, and a fresh protected evaluation.

This note records the *completed* in-distribution engineering pilot, whereas the SLOG external generalization project remains **deferred/unperformed**.
