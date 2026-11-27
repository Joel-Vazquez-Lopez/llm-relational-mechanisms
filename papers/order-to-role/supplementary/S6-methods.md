[← Supplement](README.md) · [Paper II](../README.md) · [Methods](../METHODS.md) · [Reproduce](../REPRODUCE.md)

# S6 · Methods and reproducibility

This page adds detail to [METHODS](../METHODS.md) and [REPRODUCE](../REPRODUCE.md); it does not repeat them. Links to code point to the frozen code snapshot `9c43792`.

## 1. Prompts and scoring

- **Prompt.** `Sentence: {sentence}\nQuestion: {question}\n{instruction}\nAnswer:`, wrapped as one user turn in the model's chat template, with the generation prompt appended. There is no system message, except in the OTR-15 `system` format. Qwen3.8 and Granite use `enable_thinking=False`.
- **Generation score** (C1, S2, S5). Greedy decoding with up to 40 new tokens. The reply is scored by the extractor in [`otr_generate_check.py`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/scripts/otr_generate_check.py), which credits a reply that names the expected person as its answer. Each result file records the reply kinds (`direct`, `frame`, `statement`, `inferred`, `ambiguous`, `none`) and both strict and first-name accuracies. With "Reply with one name only.", 98.8–100% of replies in the 26 November wording battery are direct names.
- **Logit margins** (OTR-03, -07, -09, -13, -14). logit(answer) − logit(distractor) at the answer position, using the first token of each name with no leading space after the chat template's assistant tag.
- **Forced choice is not used for behaviour.** Without an instruction, chat models restate the sentence, and first-token forced choice then scores the restated first name. This artefact is documented in the [historical review package](historical-packages.md#review-package-16-november).
- **Shared instrumentation.** Scoring and patching build their prompts through one function (`chat.apply_template`), so system prompt, template options and assistant prefix are always identical between them. On a tiny model, tests check that the two produce the same token ids and the same baseline margin under nine chat settings (`tests/test_scoring_patching_consistency.py`, added by the code-integration PR). Under every setting used by a completed run, the token ids are byte-identical to the code that produced the results.

## 2. Measures

| Measure | Definition | Used in |
|:--|:--|:--|
| Transplant coefficients | Copy one name's hidden state at layer *l* from a source sentence into a base sentence. Regress the change in the base answer margin on the source name's order, function and role, within base item and slot, with a family bootstrap for CIs | C2, C3, S3 §1 |
| Validity null | Role slope among transplants whose sources share their left context. Must be 0 | S3 §1 |
| Residual repair | (patched − failing) / (working − failing) margin, after replacing the failing question's residual at the answer position with the working question's. Items with a working–failing gap of at least 1.0 logit | C4, S3 §2, S5 §3 |
| Handover | First sampled layer with residual repair ≥ .5 (ties to the earlier layer) | C4, frozen layers |
| Head repair | The same ratio, after patching the outputs of *k* attention heads (`o_proj` input slices) from the working run | C5, S3 §3 |
| C5 controls | `random_same_layers`: 5 draws of 20 heads from the frozen heads' layers. `other_failing`: the frozen heads with another item's failing-run output. `random_nudge`: the failing head output plus a random vector with the norm of (working − failing) | C5 |
| C5b | The frozen heads with the item's own working-run output vs another item's working-run output | XL-C5b |
| "By" damage | Change in the working answer margin after transplanting the "by" token state from the failing (C6) or the patient (C8) question, normalised as in OTR-13 | C6, C8 |
| Share agent after prefix | Among wrong failing-question items, the share with positive agent margin after "Someone was V-ed by" | C7 |

**Bootstrap.** For confirmatory claims: 2000 resamples of name-pair families with seed 2026, and 95% percentile intervals (2.5th and 97.5th percentiles; frozen scorer). Development files carry their own run-level bootstrap. Only family-level resampling is used, because items within a family share names.

## 3. Frozen settings and their sources

Layers and heads were chosen once on development data and never re-chosen (FREEZE.md §4). The supplement script recomputes every handover from the named development files: 5 English and 10 Spanish/Galician, all matching. It also compares each model's frozen top-20 head list with the development file's ranking: all five lists are identical and in the same order. The main build script checks the 38 development files listed in [`freeze_inputs.sha256`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/freeze_inputs.sha256) against the archive: **38 of 38 match** ([`../results/MANIFEST.json`](../results/MANIFEST.json)).

Model weights are the Hugging Face snapshots listed in FREEZE.md §4 (full revision hashes there). Jobs run with the hub offline, so development and test runs use the same weights.

## 4. Compute and software

- **Hardware.** Uppsala Pelle cluster, one H100 NVL per job for the 14–32B models, two H100s with `--device-map auto` for the 70B-class models. bfloat16.
- **Software recorded in the result files.** The 11 input files that record package versions (OTR-03 runs) all show Python 3.11.18, torch 2.15.0+cu132 and transformers 5.23.0 ([`st15_software_versions.csv`](tables/st15_software_versions.csv)). Other runners record the configuration but not package versions.
- **Tests.** The code-integration PR adds two CI jobs. The *lightweight* job runs on Python 3.10 and 3.12 with pytest and NumPy. The *scientific CPU validation* job runs CPU PyTorch 2.15, transformers 5.23+, tokenizers and accelerate, and exercises scoring, patching, chat templates, device maps and every OTR runner on tiny randomly initialised models, with the hub offline. Neither downloads weights or opens an evaluation split.

## 5. Evaluation boundaries

- **Test set 0** was opened once, on 26 November: 48 English jobs at `f6ea0e1` and 40 Spanish/Galician jobs at `c40a1f7`, with no reruns. The canonical attempt logs are on the results branch.
- **Test sets 1–3 are closed.** No script in this supplement reads them. The generator counts in [S2 §4](S2-linguistic-controls.md#4-stimulus-inventory-and-design-checks) are design facts (families and items per split), not model outputs.
- The supplement reads test0 only through the frozen scorer's summaries in `../results/confirmatory/`.

## Rebuild

From the repository root, with numpy and matplotlib installed (the main results must be built first, because S1 reads their confirmatory output):

```bash
git worktree add /tmp/otr-results 25c6914e29473e4b1a1063600c48743f68b705c3   # order-to-role/results-dev
python papers/order-to-role/analysis/build_results.py --results /tmp/otr-results/results --code . \
    --out papers/order-to-role/results \
    --results-commit 25c6914e29473e4b1a1063600c48743f68b705c3 --code-commit "$(git rev-parse HEAD)"
python papers/order-to-role/supplementary/analysis/build_supplement.py --results /tmp/otr-results/results \
    --code . --main-results papers/order-to-role/results --out papers/order-to-role/supplementary \
    --results-commit 25c6914e29473e4b1a1063600c48743f68b705c3 --code-commit "$(git rev-parse HEAD)"
```

**File selection rules.** Frozen development files are named in `build_results.py` and imported. The question-wording battery uses the 26 November runs only (one run per wording and model). BX blocks, Spanish and Galician use the earliest development run with the one-name instruction. OTR-15 uses the single run per model. The 70B models use their single OTR-09 run and the OTR-03 run without the `_aq_` suffix. Swedish uses the runs with the one-name instruction.

`--code` must point to a checkout that contains the OTR code: `main` after the code-integration PR, or `order-to-role/stimuli`. The script prints the number of inputs read (133) and the number of handover mismatches (0). Every figure and table under `supplementary/` is overwritten. With the same inputs, a rebuild is byte-identical: SVG dates are stripped and element ids are seeded. Committed outputs were built from results `25c6914` and code `9c43792`.
