[← Supplement](README.md) · [Paper II](../README.md) · [How it developed](../DEVELOPMENT.md)

# Historical packages

These are earlier Order-to-Role packages, kept for provenance. They are **not current evidence**. All of them predate the freeze on 26 November and contain development results only. Some use older stimuli, older answer extraction, a narrower model set or a framing that has since been corrected. Each one is linked at an immutable commit and has not been copied. Their original titles, claim labels and numbering are kept, and their labels (for example "C1–C8" in the review package) are **not** the frozen C1–C8 of FREEZE.md.

| Package | Date | Location (pinned) | Status now |
|:--|:--|:--|:--|
| Review package | 16–17 Nov | [`docs/order_to_role/review_2026-11/`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/review_2026-11) | Historical, exploratory |
| Early cross-model figures | 16 Nov | [`docs/order_to_role/figures/`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/figures) | Superseded |
| Handoff README | 16 Nov | [`ORDER_TO_ROLE_HANDOFF.md`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/ORDER_TO_ROLE_HANDOFF.md) (moved to `docs/order_to_role/state_2026-11-16/HANDOFF.md` by the code-integration PR) | Historical working note |
| State snapshot and first draft | 24–25 Nov | [`docs/order_to_role/state_2026-11-25/`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/state_2026-11-25) | Superseded by the freeze and this section |
| 70B extension plan | 27 Nov | [`docs/order_to_role/BIG_MODELS.md`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/BIG_MODELS.md) | Current plan, exploratory ([S3 §4](S3-layerwise-and-heads.md#4-the-70b-class-models-exploratory)) |
| Earlier pretrained-model experiments | before 13 Nov | [`investigations/pretrained_llms/`](../../../investigations/pretrained_llms/README.md) | Motivation only; not Paper II evidence |

## Review package, 16 November

A reviewer-facing package written before any test set existed: research question, eight claims with status (valid, dropped or open), seven figures and eight tables, a script that regenerates them, a [review brief](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/review_2026-11/REVIEW_BRIEF.md) listing what reviewers should attack, and `tables/provenance.json`. Some of its runs were committed to the stimuli branch by mistake (commits `91e66ca`, `e1bff8c`, `e725b8f`), so reproducing it needs both branches.

| Its item | What it showed then | Status now |
|:--|:--|:--|
| f1 / t1, forced choice vs generation | Without an instruction, first-token forced choice scores restatements and produced an apparent subject/object asymmetry. OLMo-2-13B moved the other way | **Still holds** as a methods result. It is the reason every current behavioural number uses instructed generation ([S6 §1](S6-methods.md#1-prompts-and-scoring)) |
| f2 / t2, the stranded "by" question | "Who was someone V-ed by?" reverses roles in two strong models | **Replaced** by the five-model 26 November battery ([S2 §1](S2-linguistic-controls.md)) and confirmed on test0 (C1) |
| f3 / t3, within-item formats | The same items answered correctly as full sentences | Historical; the current answer-process evidence is C7 (neutral prefix) |
| f4, logit lens | Where the wrong answer is chosen (descriptive) | **Replaced** by the OTR-09/13/14 layer pipeline |
| f5 / t5, stranded-P frames | Stranding in general is not the cause; the filler-gap account was dropped | A negative result that still stands; not re-run with the current extractor |
| f6 / t6 / t8, passive agent questions in three-place frames (by verb; name vs sentence answers) | At the time, mostly fine, with lexically and structurally conditioned errors (open) | Historical; not re-run. The current three-place evidence is the BX ditransitive block ([S2 §2](S2-linguistic-controls.md#2-broader-english-constructions-bx-blocks)) |
| f7 / t7, question variants by format | Mixed effects of wording | **Replaced** by S2 §1 and OTR-15 (S2 §3) |
| "C5", shared processing sequence across 9 models | Valid for Gemma and Qwen3.8 at the time | The five-model, frozen-layer version is FREEZE.md C4 |
| legacy_* figures | Copies of the 16 November cross-model figures | Superseded (next section) |

## Early cross-model figures, 16 November

Four PNG figures and `layer_summary.csv`, built from OTR-03/04 development runs. Part of the cross-model figure script **copies numbers from the handoff README rather than reading result files**. These figures used OTR-04 (question-side flips) for the "where is the question read" story, which the OTR-09/13/14 pipeline has since replaced, and they cover an earlier model set (Qwen2.5, OLMo-2, Gemma 4, Qwen3.8), not the final five. They are kept for history only. The current versions are [fig2–fig4](../RESULTS.md#3-from-storage-to-answer-the-layer-pipeline) and [sfig05–sfig07](S3-layerwise-and-heads.md).

## Handoff README, 16 November

A working note for the next chat session: state of the study, patches still to apply, commands, rules (for example, no test set before `--test-set` flags existed) and the next steps. It documents the second generation-check round that led to the instructed-generation standard. It is a process record, not a results source.

## State snapshot and first draft, 24–25 November

| File | What it is | Caveat |
|:--|:--|:--|
| `paper_draft.md` | First full draft (24 Nov) | It says the names carry role "more strongly than their position or grammatical function". That comparison of sizes holds, but the draft does not report that order/focus carries a real signal in three models. Its psych-verb sentence rests on transcribed development numbers. The corrected framing is in the [evidence ledger](../EVIDENCE.md#6-corrections-and-disclosures). Gaps marked `[test0]` are now filled by S1 |
| `results_log.md` | Development results log, 20–24 Nov | Some numbers (for example the psych-verb coefficients quoted in [RESULTS §2](../RESULTS.md#2-the-names-carry-their-thematic-role)) appear only here and are transcribed, not recomputed |
| `FREEZE_draft.md` | Draft pre-registration | **Not the freeze.** The frozen texts are FREEZE.md (`f6ea0e1`) and FREEZE_XL.md (`3811d7c`, `c40a1f7`) |
| `result_index.md` | Which file backs which claim (draft) | Superseded by the manifests of `build_results.py` and `build_supplement.py` |
| `TODO.md` | Task list at the time | Historical |

## Not in the repository

The programme's presentations, posters and supervisor updates are kept in the Research Communications collection. Planning notes kept outside the repository (for example in the project's shared documents) are not evidence and are not linked from here.
