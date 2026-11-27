# Order-to-Role results archive (results-dev @ 25c6914)

This package preserves, byte for byte, every file that branch `order-to-role/results-dev` adds to `main`: 458 files under `results/`, 206.4 MB, plus the one superseded file version that exists only in that branch's history. The 458 files were introduced by 13 commits between 22 and 27 November 2026. It is the immutable reference that must be verified before that branch can be retired. The files are the raw outputs behind Paper II (*Order-to-Role*); the curated, citable summaries are on `main` in `papers/order-to-role/`.

| Item | Value |
|:--|:--|
| Source repository | https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms |
| Source branch and commit | `order-to-role/results-dev` @ `25c6914e29473e4b1a1063600c48743f68b705c3` |
| Branch point on `main` | `c3e07b380428d73d72b9172e9dd262c1216dff23` (PR #12's results snapshot) |
| Compared against `main` | `dde717dd1873692915a733b77c10be71386a68bb` (after PRs #19–#21) |
| Inventory | `inventory.csv`: one row per file, with source commit, type, experiment, model, split, material, role in the record, evidence status, identical copy on `main`, verification need, frozen-evaluation and third-party flags, redistribution terms, destination |
| Superseded versions | `history.csv`: file versions that were replaced on results-dev and are reachable from no other branch (one). Every earlier results/ version on `order-to-role/stimuli` is already in `main`'s history through the PR #19 merge |
| Checksums | `SHA256SUMS` (sha256 of every file); the archive file has its own `.sha256` |
| Builder | `papers/order-to-role/archive/make_archive.py build` (deterministic) and `verify` |

## What is inside

| Group | Files | Evidence status | Where it lives |
|:--|--:|:--|:--|
| test0 outputs read by the frozen scorer (88 jobs: 48 English at `f6ea0e1`, 40 Spanish/Galician at `c40a1f7`) | 88 JSON | Confirmatory | Git (`results/`) and archive |
| test0 attempt logs (`test0_attempts_pelle.md` is the canonical log) | 2 MD | Confirmatory provenance | Git and archive |
| Slurm logs of the 88 test0 jobs | 88 OUT | Confirmatory provenance | Git and archive |
| Development files the frozen layers and heads were read from (`freeze_inputs.sha256`) | 20 JSON | Development, frozen-setting source | Git and archive |
| Other development outputs read by the Paper II build scripts | 50 JSON | Development | Git and archive |
| Development outputs not read by the build scripts (WIKI v0.1 and per-item reruns, C5 dev controls, OTR-13 BX and second-working-question checks, OTR-14 BX) | 36 JSON | Development | Git and archive |
| 70B-class development runs (Llama 3.3 70B OTR-03 and OTR-09, Qwen2.5 72B OTR-09) | 4 JSON | Exploratory, no freeze | Git and archive |
| Other run logs (development runs, downloads, WIKI extraction, 70B smoke tests) | 77 OUT | Provenance / engineering | Git and archive |
| Per-run diagnostic plots, regenerable from the JSON beside them | 87 PNG | Same as their JSON | Git and archive |
| WIKI-ROLES extraction summaries | 2 JSON | Development input | Git and archive |
| **WIKI-ROLES item sets and check sheets (Wikipedia text)** | 2 JSONL, 2 CSV | Development input | **Archive only** until the licence decision below |
| Superseded version that exists only in results-dev history: the first, partial `wiki_roles_en.jsonl` (2657 sentences, commit `9cca09e`, replaced by the full file in `f4cad6a`; no archived run is recorded against it) | 1 JSONL, under `history/9cca09e/` | Development input (superseded) | **Archive only** (Wikipedia text) |

The JSON outputs of the generation check (`otrgen_*`) and OTR-15 contain the prompts and model replies. The prompts are the study's own generated template sentences. The OTR-03/07/09/13/14 outputs and the WIKI/UD OTR-09 outputs contain no third-party text (only names, labels and numbers). No model weights, tokenizers or test sets 1–3 are included. No results-dev file belongs to test sets 1–3.

## Licences and attribution

- **Research outputs** (all JSON, PNG, OUT and MD files) are the authors' own outputs: model generations and measurements on generated stimuli. **The repository does not yet declare a licence**, so reuse terms for these files are undefined until the owner adds one. This is the main open blocker (see `README.md` in this folder).
- **Model outputs** come from Gemma 4 31B-it, Qwen3.8-27B, phi-4, OLMo-3.1-32B-Instruct, Granite-4.2-30B, Llama-3.3-70B-Instruct and Qwen2.5-72B-Instruct, at the Hugging Face revisions listed in `docs/order_to_role/FREEZE.md` §4 and `docs/order_to_role/BIG_MODELS.md`. Their use is subject to each model's licence and use policy. No model files are redistributed.
- **Wikipedia text** (`results/wiki_roles_en.jsonl`, `results/wiki_roles_v2_en.jsonl`, `results/wiki_check_sheet.csv`, `results/wiki_roles_v2_en_check_sheet.csv`): sentences from English Wikipedia, dump `20231101.en` via the Hugging Face dataset `wikimedia/wikipedia`. Wikipedia text is licensed **CC BY-SA 4.0**. Each sentence carries its article title and article id, which is the attribution. Any redistribution must keep that attribution, state the licence and share alike. These four files are therefore kept out of Git until the owner decides how to license and attribute them in the repository.
- **Universal Dependencies** sentences are not in this archive. They are in `src/relational_lab/experiments/ud_en_natural.json` on `main` (see `THIRD_PARTY_DATA.md` there).

## Software and data provenance

- Runs: Uppsala Pelle cluster, H100 NVL GPUs, bfloat16, Hugging Face hub offline. The 42 result files that record package versions all show Python 3.11.18, torch 2.15.0+cu132 and transformers 5.23.0.
- Code: the scripts and generators that produced these files are on `main` with their history, merged from `order-to-role/stimuli` @ `9c43792` in PR #19. Each file's generator and runner version is recorded inside it.
- Frozen protocols: `docs/order_to_role/FREEZE.md` (commit `f6ea0e1`), `FREEZE_XL.md` (`3811d7c`, native check `c40a1f7`), scorer `scripts/otr_freeze_score.py` v1.1.0.

## Retrieval

From Git, until the branch is retired:

```bash
git fetch origin order-to-role/results-dev
git archive 25c6914e29473e4b1a1063600c48743f68b705c3 results/ | tar -x -C /tmp/otr
cd /tmp/otr && sha256sum -c <path>/SHA256SUMS
```

From the package:

```bash
sha256sum -c order-to-role-results-dev-25c6914.tar.gz.sha256
python papers/order-to-role/archive/make_archive.py verify --archive order-to-role-results-dev-25c6914.tar.gz
```

Once the package is deposited (proposed: a Zenodo record or a GitHub release asset, with a DOI), its URL and checksum belong in `papers/order-to-role/REPRODUCE.md`.
