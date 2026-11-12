# Prospective SLOG External-Validation Protocol — Development Stage

**Created:** 12 November 2026. **Status:** DEFERRED FUTURE EXTENSION / NO MODEL OR TEST DATA EVALUATED. This document is **not frozen**. We completed public-development eligibility and prompt-preview checks, but will not run Qwen or other pretrained models in the current paper's final experiment phase. The independent SLOG generalization/test archive remains unopened; no external model results are claimed.

## Objective and what this experiment can establish

Test whether the **behavioral taxonomy** of complementary semantic-role answers—correct, reversed, and collapsed—is useful outside the existing synthetic two-event grammar, using an independently authored benchmark and pretrained Transformer language models.

This is an external *error-pattern* study. It **does not** replicate the causal impact of targeted scratch-model exposure, nor prove internal role binding.

## Benchmark

- Official source: https://github.com/bingzhilee/SLOG
- Li et al. (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.194/
- First permitted input for preflight: **public `data/cogs_LF/dev.tsv` only**, encoding `sentence<TAB>logical_form<TAB>category`. This public development file contains 3,000 `in_distribution` rows and 1,046 rows spanning six labeled structural categories; these labels are retained for stratified feasibility auditing, not rejected. The output LF includes `verb.agent(event, argument)` and `verb.theme(event, argument)` clauses. The predicate variable index anchors the verb in the source sentence (zero-based).
- **No access to `generalization_sets.zip`**, SLOG test, or other held-out data until the protocol is checked, the model/checkpoint revisions and inference scoring are fixed, and a run is authorized.
- Pretrained-model prompt selection, scoring calibrations, and competence gating may use **only dev rows labeled `in_distribution`**, with additional disjointness exclusions if needed. The six structural-category slices of public dev are examined for feasibility only until the protocol is frozen; do not tune on their observed model performance.
- SLOG's `theme` role is not perfectly equivalent to the scratch grammar's `patient`, so make any comparison explicit and limited.

## Eligibility rule (development stage)

Only consider a sentence if its LF describes at least two predicate events. Select a predicate/role pair if:
1. It has exactly one extractable `agent` and one `theme` argument.
2. Both arguments map to distinct unique word-head mentions (proper names or unambiguous noun heads) in the original sentence.
3. Its predicate variable aligns to a unique verb-token mention in the sentence.
4. No ambiguous coreference, repeated answer surfaces, or repeated predicate surfaces are admitted.

These conservative rules sacrifice yield for semantic reliability; an individual event may have legitimate richer arguments (e.g. noun phrases) that this extraction does not preserve. At least two full eligible events with four disjoint participant labels are required for any **four-choice** event-selection extension. Do not infer four-choice coverage merely from multi-event sentence count.

A preliminary programmatic audit of the public dev annotations found **4,046 sentences**, of which approximately **812** have an eligible role-pair in a sentence containing two or more annotated predicates; approximately **462** have two or more such eligible events and only approximately **53** satisfy the stronger four-distinct-participant condition. These are **development-split feasibility counts, not performance**; the separately versioned Python preflight supplies the auditable counts and hashes.

## Stage 0 — data preflight [READY; no inference]

Use `scripts/paper_slog_preflight.py` from the paper branch. The script:
- Reads only a file named `cogs_LF/dev.tsv`.
- Checks role alignment and rejects ambiguous surface mentions.
- Measures total eligible cases, two-event coverage, and approximate syntactic strata.
- Writes only counts, rejection reasons, and input/script SHA256, never held-out examples or model predictions.
- Refuses to overwrite an existing report.

**Go/no-go** before model selection: at least 500 distinct multi-predicate sentences with an extractable role pair, and a documented manual/independent audit of annotation mapping on public development examples. If the mapping is invalid, mark stage failed; don't inspect protected SLOG material to repair it.

## Stage 0b — QA prompt and semantic projection review [IN PROGRESS]

The user-run development label audit completed with the same official dataset SHA256, **50 unique selected sentences**, an available macOS MPS backend, Python 3.13.13, PyTorch 2.15.0 and Transformers 5.20.2. Printed examples looked structurally aligned, but all 50 selected sentences and every eligible verb must be reviewed before calling the projection validated.

`scripts/paper_slog_prompt_preview.py` locally constructs four balanced **development-only** question variants per eligible event (agent/theme × A/B option order). Each marks the exact source verb with `[EVENT]` tags, includes the two candidate *surface heads*, and writes a local JSONL preview with gold labels **for human verification only**. It does **not** load a model or any protected split. The candidate prompt wording is *provisional*, not frozen or validated for model competence. Do not commit raw source/gold preview text.

Stop if any incorrect role alignment, invalid head reference, missing event disambiguation, or semantic mismatch is found. Corrections to extraction/prompting must be documented and followed by a new public-dev preflight **before** freeze. A sample audit provides diagnostic confidence, not a proof of complete correctness.

## Stage 0c — local pretrained checkpoint engineering calibration [SCRIPT READY; NOT RUN]

Local machine environment reported **16 GiB unified memory** (`hw.memsize=17179869184`), Apple Silicon MPS available, Python 3.13.13, torch 2.15.0, Transformers 5.20.2.

A feasible provisional checkpoint for this engineering gate is `Qwen/Qwen2.5-1.5B-Instruct` (official Hugging Face model card; Apache-2.0, 1.54B parameters). It is not yet the frozen protected-evaluation model. The script `scripts/paper_slog_dev_model_calibration.py` deterministically selects **16 in_distribution public-development sentences** with at least two annotated predicates and exactly one eligible event per sentence, verifies the audited dataset SHA256, and scores **agent/theme × balanced A/B candidate order** with a chat-template next-token comparison. This is a **model-inference engineering check**, not an external generalization result; its single run is logged with the resolved model commit SHA, script hash, PyTorch/Transformers versions, and per-example scoring.

Before any protected generalization data are considered, check that option A/B each correspond to a single tokenizer token at the model's answer position; that both-order answer consistency is high; and that the model can correctly answer development cases. The code currently reports the first ordering separately; final role-assignment results must use a separately specified order-robust aggregation. Any prompt tuning or model replacement using development observations must be disclosed and the **final** protocol independently frozen; do not select scoring variants based on protected outcomes.

A genuine role-projection audit of all 50 chosen public-dev sentences is still **outstanding**. Passing the inference gate alone does not substitute for it.

## Stage 1 — protocol freeze [NOT DONE]

Before model inference, freeze in a new file:
- Semantic projection: exact agent/theme questions; verb identifier; how candidate names map back to surface spans; rejection and ambiguity rules.
- Model architecture and **exact revision hashes**: at least one feasible off-the-shelf instruction-tuned pretrained causal Transformer (candidate Qwen2.5-1.5B-Instruct) and, resources permitting **before seeing outcomes**, a larger fixed checkpoint (candidate Qwen2.5-7B-Instruct). Model choice remains provisional until hardware feasibility and licensing checks; do not swap models in reaction to evaluation scores.
- Prompts: natural-language question about the specific event, two response options corresponding to agent and theme. Both answer-choice permutations are scored to quantify position bias. No revealed gold during prompt tuning. Prompt template must be fixed in advance.
- Primary outcome: event-level correct / reversed / same-person (collapsed) assignment rates under forced two-candidate scoring; denominator is unique sentence × predicate, and uncertainty should cluster by sentence and report model-stratum variation.
- Baselines and controls: candidate-order bias, swapped question labels, role-blind always-option baselines, in-domain competence on a separate public subset, distractor event substitution and four-choice event selection only when four distinct participants exist. Do not pool pairs from multiple questions as independent samples.
- Generalization categories and exclusions: prespecified; use published category labels only after protocol freeze on protected items.
- Competence gate: measurable, chosen on development calibration **before** protected results. If no model meets the gate, report a failed competence/preflight result and do not reinterpret unrelated patterns as evidence for transfer.
- Budget and run limit: at most one prespecified held-out evaluation per model with immutable reports; no prompt/site/cherry-picking after the protected run.

## Stage 2 — prospective model evaluation [NOT DONE]

Evaluate candidate-option likelihoods using a fixed chat template and exact tokenizer settings. A two-option task measures role selection **conditioned on the known event**, not native open-ended answer generation. Optionally report unconstrained generation as secondary with a fixed parser and nonanswers counted as errors; do not silently conflate these scoring regimes.

Under four-choice scoring, analyze event choice and role choice separately only in the subset with two annotated, disjoint, identifiable events. Do not generalize a two-option result into a claim of multi-event selection.

Report:
- Dataset and training-contamination limitations.
- Eligibility yield per SLOG generalization type.
- Model-by-type accuracy and event-level confusion partition.
- Role asymmetry and answer-position preference.
- Confidence intervals clustered at least by sentence; distinct model checkpoints are not independent datasets.
- Negative or null findings unchanged.

## Stopping rules

- No new ML-101 or additional scratch-trained mechanism search.
- One external-validation protocol, no hidden iteration on held-out examples.
- If Stage 0 fails, stop and explain the limitation in the manuscript.
- If structural held-out yield is insufficient, report the feasibility failure and reconsider the benchmark only as a *new registered study*, not as an unreported replacement.

## Reproducibility

Do not commit third-party raw SLOG dataset copies or model weights. Commit the audit script and the counts/hash report only. Store per-example data or model outputs locally until copyright/license and privacy review. Pin dataset revision (Git commit or blob SHA) and model revision when frozen.

**Current status:** Manuscript sections 3.1–3.5 saved in `paper/manuscript.md`. Preflight code committed; no trained-model inference and no protected SLOG generalization data opened.

## Decision checkpoint — 12 November 2026

Following a completed five-cohort audit of the already archived scratch-Transformer protected results, **the current manuscript's experimental work is closed**. The external pretrained-model SLOG validation is postponed to a separately registered future study. The public dataset preflight and question-preview scripts remain available, but the `paper_slog_dev_model_calibration.py` model-inference engineering step has **not** been executed. Do not interpret the public-development sample review as model evaluation or as evidence of natural-language generalization.
