[← Order-to-Role](README.md) · [Experiment map](EXPERIMENTS.md) · [Integration](INTEGRATION.md)

# Reproducing Order-to-Role

There are three levels of reproduction, from cheapest to most expensive:

1. **Rebuild every table and figure** on these pages from the archived result files. This needs no GPU and takes a few minutes.
2. **Re-score the frozen evaluation** from the archived test0 outputs with the frozen scorer. This needs no GPU and opens no new test data.
3. **Rerun the experiments** on models. This needs GPUs (one H100 per job for the 14–32B models, two for 70B).

## Where the code and results are

| What | Where (until the [integration](INTEGRATION.md) is merged) | Pinned commit |
|:--|:--|:--|
| OTR scripts, generators, freezes, scorer, tests | [`order-to-role/stimuli`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/stimuli) | `9c437928` |
| Development, test0 and 70B result files | [`order-to-role/results-dev`](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/tree/order-to-role/results-dev), folder `results/` | `25c6914e` |
| This section's tables, figures and scorer output | `papers/order-to-role/results/` on `main` | – |

Once the code integration is merged, the scripts are on `main`; only the result files stay on the results branch.

## 1. Rebuild the results section

```bash
git clone https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms.git
cd llm-relational-mechanisms
git fetch origin order-to-role/results-dev order-to-role/stimuli
git worktree add ../otr-results 25c6914e29473e4b1a1063600c48743f68b705c3
git worktree add ../otr-code 9c437928f61351a72e8d7be4cac6d3a964844a7e
python -m pip install numpy matplotlib
python papers/order-to-role/analysis/build_results.py --results ../otr-results/results --code ../otr-code --out papers/order-to-role/results --results-commit 25c6914e29473e4b1a1063600c48743f68b705c3 --code-commit 9c437928f61351a72e8d7be4cac6d3a964844a7e
git diff --stat papers/order-to-role/results
```

After the code integration is merged, `--code .` replaces `--code ../otr-code`.

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

## 2. Re-score the frozen evaluation only

```bash
python ../otr-code/scripts/otr_freeze_score.py --results ../otr-results/results --split test0
python ../otr-code/scripts/otr_freeze_score.py --results ../otr-results/results --split test0 --stimuli bx-transitive
python ../otr-code/scripts/otr_freeze_score.py --results ../otr-results/results --split test0 --stimuli es
python ../otr-code/scripts/otr_freeze_score.py --results ../otr-results/results --split test0 --stimuli gl
```

The scorer reads only result files that already exist. Do not run any `--split test0` experiment again, do not open test sets 1–3, and do not change thresholds, layers or heads after seeing results. Any new analysis of test0 outputs is exploratory and must be labelled as such.

## 3. Lightweight tests

```bash
python -m pip install -e '.[dev]' numpy
PYTHONPATH=src python -m pytest -q tests
```

On a CPU-only machine with the integrated code, this gives 147 passed and 57 skipped. The skipped tests need `torch` or a GPU.

## 4. Rerunning experiments

Each experiment is a script in `scripts/` (see the [experiment map](EXPERIMENTS.md)) with a matching Slurm job in `scripts/slurm/`. One-time cluster setup is `scripts/slurm/setup_pelle.sh`. Every result file records:

- the model and its Hugging Face revision
- the generator version, seed and split
- the chat-template setting and instruction
- the script hash and library versions

Record the same for any new run. The 70B models use `--device-map auto` over two GPUs ([plan](https://github.com/Joel-Vazquez-Lopez/llm-relational-mechanisms/blob/9c437928f61351a72e8d7be4cac6d3a964844a7e/docs/order_to_role/BIG_MODELS.md)). They are exploratory until a separate freeze addendum is committed.

## 5. Validity checks to keep

- Score behaviour on generated answers, not first-token forced choice.
- Report repair as a conditional share of the margin gap, never as accuracy.
- Keep the left-context validity check in OTR-03 at zero.
- Draw test-run donors only from the run's own split.
- Report every model and claim, including failed and not-evaluable cells.
