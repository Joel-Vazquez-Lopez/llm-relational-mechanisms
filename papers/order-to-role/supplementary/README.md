[← Paper II](../README.md) · [Results](../RESULTS.md) · [Evidence ledger](../EVIDENCE.md) · [Methods](../METHODS.md)

# Paper II · Scientific supplementary materials

**Working supplement, 27 November 2026.** This collection extends the [Paper II results](../RESULTS.md) with full statistical tables, extended layerwise and head analyses, model-by-model results, linguistic controls, multilingual evidence and method details. Like the main results, every figure and table is rebuilt from archived result files by one script, [`analysis/build_supplement.py`](analysis/build_supplement.py). The source commits and the checksum of each of its 138 input files are recorded in [`MANIFEST.json`](MANIFEST.json).

This is not yet a submission supplement, and its sections are not numbered as manuscript supplements. Presentations, posters and supervisor updates are kept in the programme-wide Research Communications collection, not here.

## Sections

| Section | What it adds to the main results | Main figure |
|:--|:--|:--|
| [S1 · Statistics of the confirmatory evaluation](S1-statistics.md) | Every frozen claim on test set 0, with estimate, 95% CI, sample sizes and secondary fields, for English, BX, Spanish and Galician | [sfig01](figures/sfig01_confirmatory_estimates.svg) |
| [S2 · Linguistic controls](S2-linguistic-controls.md) | Nine question wordings, the failure by construction, ditransitive and psych-verb blocks, prompt formats, stimulus inventory | [sfig02](figures/sfig02_question_wording.svg), [sfig03](figures/sfig03_bx_blocks.svg), [sfig04](figures/sfig04_prompt_formats.svg) |
| [S3 · Layerwise and head analyses](S3-layerwise-and-heads.md) | Role and order codes with confidence bands (core vs BX), repair by construction, head counts and head locations, 70B curves, head knockout and the averaged direction (OTR-12) | [sfig05](figures/sfig05_role_code_core_vs_bx.svg)–[sfig08](figures/sfig08_big_models.svg), [sfig10](figures/sfig10_knockout.svg) |
| [S4 · Model-by-model results](S4-models.md) | One card per model: frozen layers, behaviour, role code, heads, transfer, verdicts, known issues | — |
| [S5 · Multilingual evidence](S5-multilingual.md) | Spanish and Galician behaviour by construction, handover by language, the Swedish pilot | [sfig09](figures/sfig09_handover_by_language.svg) |
| [S6 · Methods and reproducibility](S6-methods.md) | Prompts, measures, bootstrap, frozen settings, input checksums, software, how to rebuild | — |
| [Historical packages](historical-packages.md) | Earlier review packages, state snapshots and drafts, what each one still shows and what has replaced it | — |

## Evidence status

Each item carries one of three labels, and the labels are never pooled:

| Label | Meaning | Where |
|:--|:--|:--|
| **Confirmatory** | Frozen claims, rules, layers and heads, committed before test set 0 was opened (`f6ea0e1`, Spanish/Galician `3811d7c`, native check `c40a1f7`), scored once by the frozen scorer v1.1.0 | S1, and the verdict columns of S4 |
| **Development (exploratory)** | Development split (41 of 276 name-pair families). Layers and heads were chosen on these data, so they describe the data rather than test a claim | S2, S3, S4, S5, unless marked otherwise |
| **Historical** | Packages written before the freeze, sometimes with older stimuli, extractors or wording. Kept for provenance, not as current evidence | [Historical packages](historical-packages.md) |

Two inputs carry extra caveats. The Swedish stimuli have not been checked by a native speaker (the check sheet is still blank). The 70B models have no frozen settings.

## Provenance

| Source | Commit | Role |
|:--|:--|:--|
| Result files: `results/` on `main` (originally `order-to-role/results-dev`) | `25c6914` | All result JSONs read by the script; [inventory and archive](../archive/README.md) |
| OTR code on `main` (originally `order-to-role/stimuli`) | `9c43792` | Generators (stimulus counts), frozen scorer, FREEZE files |
| Main results, [`../results/`](../results/) | this branch | The frozen scorer's test0 summaries (`confirmatory/test0_*.json`), read here rather than recomputed |

Rebuild steps are in [S6](S6-methods.md#rebuild). The script loads no model, runs no experiment and reads no test-split output beyond the frozen scorer's existing test0 summaries.
