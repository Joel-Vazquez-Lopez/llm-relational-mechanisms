# CORE-DIVERSITY-v1: frozen prospective run instructions

**Current state:** source-generator feasibility has passed on GitHub Actions. The training runner includes its own three-layer Transformer and native scorer so it does not depend on legacy scripts absent from a fresh GitHub checkout. These commands use only new prospective model seeds; no old dev/reserved/protected/L+I examples.

## On the original Apple MPS machine

Start in the existing checkout, with the original Python/PyTorch environment activated, and preserve unrelated local changes.

```bash
git status --short
git fetch origin
git switch --track origin/design/position-cb-core-diversity-v1
```

If the named branch already exists locally, use `git switch design/position-cb-core-diversity-v1` instead. Keep the old checkpoint files local and untouched.

**Before any scientific training, prepare and test the exact source and study schedule.** This may generate small JSON manifest files, but performs NO new study model inference:

```bash
python3 scripts/core_diversity_v1_preflight.py --phase prepare
python3 scripts/core_diversity_v1.py --phase prepare
python3 scripts/core_diversity_v1.py --phase smoke --device mps
```

The smoke uses **throwaway random model weights and TRAIN source examples only**, checking one optimization step and the native metric decoder. It does not score heldout test fixtures.

**Freeze these exact bytes in Git before running any new seed:**

```bash
git add -f reports/core_diversity_v1/source_manifest.json reports/core_diversity_v1/study_manifest.json
git commit -m "research: freeze CORE-DIVERSITY v1 source and schedule before training"
git push -u origin design/position-cb-core-diversity-v1
```

The runner checks the exact committed content of its own code, the source-only preflight, the launcher, the protocol, and both manifests before allowing a seed to train. Any mismatches are a STOP condition, not an instruction to change the protocol after viewing data.

## Launch the prospective study

```bash
caffeinate -dimsu bash scripts/run_core_diversity_v1.sh
```

Fixed grid: **16 independent new seeds × 4 paired arms = 64 scratch-trained models**. The launcher resumes by skipping seeds with both existing sealed JSON and existing local checkpoint. If only one exists, it stops rather than silently overwriting it. Individual seed runs write a JSON outcome and a local .pt checkpoint only at the fixed endpoint.

**Do not inspect or interpret the per-seed heldout scores until all 16 seeds have completed.** Do not change seeds, steps, training conditions, thresholds, or model selection based on partial outcomes. The final summary is written only when all reports exist.

Then commit only the small JSON reports, never model weights:

```bash
git add -f reports/core_diversity_v1/seed*_mps_step6400.json reports/core_diversity_v1/analysis_summary.json
git commit -m "study: archive complete CORE-DIVERSITY v1 results"
git push
```

## Frozen science questions

- **Primary:** Does exposing eight balanced role assignments per training quartet improve strict all-four-query accuracy on the other **16 never-trained role assignments** compared with training on one original assignment?
- **Gated secondary:** If primary succeeds, does the benefit transfer to **60 withheld name quartets**, each evaluated over all 24 role assignments?
- **Error composition:** How do correct/reversed/collapsed/wrong-event pairs shift? These analyses are exploratory.

A passed primary requires the **two-sided 95% paired-seed t interval lower bound positive**, **mean lift at least +10 percentage points**, and **12/16 seeds positive**. The gated secondary also requires a positive CI lower limit.

No new grammatical family is tested. Even positive results do not establish a unique semantic-role neural circuit.
