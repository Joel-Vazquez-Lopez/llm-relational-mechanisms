[← Order-to-Role](README.md) · [Experiment map](EXPERIMENTS.md) · [Integration](INTEGRATION.md)

# Reproducing Order-to-Role

There are three levels of reproduction, from cheapest to most expensive:

1. **Rebuild every table and figure** on these pages from the archived result files. This needs no GPU and takes a few minutes.
2. **Re-score the frozen evaluation** from the archived test0 outputs with the frozen scorer. This needs no GPU and opens no new test data.
3. **Rerun the experiments** on models. This needs GPUs (one H100 per job for the 14–32B models, two for 70B).

## Where the code and results are

Everything is on `main`.

| What | Where |
|:--|:--|
| OTR scripts, generators, frozen protocols, scorer, tests | `scripts/`, `src/relational_lab/`, `docs/order_to_role/`, `tests/`. Merged from `order-to-role/stimuli` @ `9c43792` with its history (PR #19); byte-identical to that snapshot apart from the relocated handoff note |
| Development, test0 and 70B result files | `results/`. PR #12 brought the outputs up to 21 November, and PR #25 imported the remaining authors' outputs from `order-to-role/results-dev` @ `25c6914`, except four files of Wikipedia text |
| The complete results-dev record, Wikipedia item sets included | The inventory and verified archive-building instructions in [`archive/README.md`](archive/README.md), including the four excluded Wikipedia files; external deposit remains pending |
| This section's tables, figures and scorer output | `papers/order-to-role/results/` and `papers/order-to-role/supplementary/` |

The pinned links to `9c43792` throughout these pages stay valid, and every linked file is byte-identical on `main`.

## 1. Rebuild the results section

```bash
git clone https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms.git
cd llm-relational-mechanisms
python -m pip install numpy matplotlib
python papers/order-to-role/analysis/build_results.py --results results --code . --out papers/order-to-role/results --results-commit 25c6914e29473e4b1a1063600c48743f68b705c3 --code-commit 9c437928f61351a72e8d7be4cac6d3a964844a7e
git diff --stat papers/order-to-role/results
```

Run on `main`'s own `results/`, the script reproduces the committed tables, confirmatory output and manifest byte for byte; the supplement script does the same for its figures and tables. The `--results-commit` and `--code-commit` flags only label the manifest with the snapshots the committed outputs were built from. To rebuild from the original branches instead, use `git worktree add ../otr-results 25c6914e29473e4b1a1063600c48743f68b705c3` and pass `--results ../otr-results/results`.

**What the script does:**

- Checks the 38 development files that the frozen settings were read from against [`freeze_inputs.sha256`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/freeze_inputs.sha256).
- Writes the development tables (`results/tables/*_dev.csv`, `layer_curves_dev.json`) and the five figures.
- Runs the frozen scorer on test0 for English, BX, Spanish and Galician (`results/confirmatory/`, `confirmatory_test0.csv`).
- Records the checksum of every file it read in `results/MANIFEST.json`.

Identical inputs give identical tables. Figures can differ in font rendering only.

**File selection rules** (so that a re-run with more result files does not silently pick something else):

- Frozen development files are named explicitly in the script, from FREEZE.md §4.
- BX, Spanish, Galician and UD use the single development run per model.
- WIKI-ROLES uses the earliest v0.2 run that includes the full head scan.

### Supplementary materials

`supplementary/analysis/build_supplement.py` rebuilds every supplementary figure and table from the same two inputs, and reads the confirmatory output above. Commands and file-selection rules are in [S6 · Methods and reproducibility](supplementary/S6-methods.md#rebuild).

## 2. Re-score the frozen evaluation only

```bash
python scripts/otr_freeze_score.py --results results --split test0
python scripts/otr_freeze_score.py --results results --split test0 --stimuli bx-transitive
python scripts/otr_freeze_score.py --results results --split test0 --stimuli es
python scripts/otr_freeze_score.py --results results --split test0 --stimuli gl
```

The scorer reads only result files that already exist. Do not run any `--split test0` experiment again, do not open test sets 1–3, and do not change thresholds, layers or heads after seeing results. Any new analysis of test0 outputs is exploratory and must be labelled as such.

## 3. Lightweight tests

```bash
python -m pip install -e '.[dev]' numpy
PYTHONPATH=src python -m pytest -q tests
```

This is CI's *lightweight* job: 147 passed, 61 skipped, 0 failed. The skipped tests need PyTorch. CI's *scientific CPU validation* job also installs CPU PyTorch, Transformers, Tokenizers, Accelerate and matplotlib, and runs the same suite on tiny random models with the Hugging Face hub offline: 214 passed, 3 skipped (spaCy parser), 0 failed (see `.github/workflows/ci.yml`).

## 4. Rerunning experiments

Each experiment is a script in `scripts/` (see the [experiment map](EXPERIMENTS.md)) with a matching Slurm job in `scripts/slurm/`. One-time cluster setup is `scripts/slurm/setup_pelle.sh`. Every result file records:

- the model
- the generator version and split
- the chat-template setting and instruction

OTR-03 files also record the seed, model revision field, script hash and library versions. The model snapshots used are listed in FREEZE.md §4.

Record the same for any new run. The 70B models use `--device-map auto` over two GPUs ([plan](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/BIG_MODELS.md)). They are exploratory until a separate freeze addendum is committed.

## 5. Validity checks to keep

- Score behaviour on generated answers, not first-token forced choice.
- Report repair as a conditional share of the margin gap, never as accuracy.
- Keep the left-context validity check in OTR-03 at zero.
- Draw test-run donors only from the run's own split.
- Report every model and claim, including failed and not-evaluable cells.
