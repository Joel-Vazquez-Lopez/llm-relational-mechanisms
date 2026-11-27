# Cross-surface v0 — local read-only diagnostic execution

**No training, no new checkpoints, no paper-submission work.** Uses the already-trained 12-seed ×4-arm checkpoints from TRAIN-SURFACE-FACTORIAL-v0.

## From your existing repository

\`\`\`bash
git status --short
git fetch origin
git switch --track origin/analysis/train-surface-cross-eval-v0
\`\`\`

If the branch already exists locally, use \`git switch analysis/train-surface-cross-eval-v0\`. Never force-switch over edits or stage unrelated untracked folders.

## Preflight and throwaway random-weight smoke

\`\`\`bash
python3 scripts/train_surface_cross_eval_v0_preflight.py --phase prepare
python3 scripts/train_surface_cross_eval_v0.py --phase smoke --device mps
\`\`\`

These run source-only checks and a tiny random model on **original familiar TRAIN cores only**. They do not evaluate the new role-permutation diagnostic or load old checkpoints.

## Commit the diagnostic source manifest before opening trained models

\`\`\`bash
git add -f reports/train_surface_cross_eval_v0/source_manifest.json
git commit -m "research: freeze read-only cross-surface diagnostic sources"
git push -u origin analysis/train-surface-cross-eval-v0
\`\`\`

Now run the fixed read-only diagnostic over all 12 original seeds:

\`\`\`bash
caffeinate -dimsu bash scripts/run_train_surface_cross_eval_v0.sh
\`\`\`

Expected checkpoint path for each seed: \`results/train_surface_factorial_v0/seed14901_mps_step6400.pt\` (and likewise all other 149xx seeds). If checkpoints live only in the earlier local worktree, use that same worktree or copy the corresponding .pt files **without changing their contents**. Never retrain these models.

All four arms are scored on exactly matched semantic examples across bare, frame, adjunct and combined test surfaces; original training-core source variants are also scored. The saved seed JSONs contain only aggregates and paired transitions, not raw prompts or model weights. Do not peek at partial aggregates or change the source selection after any run.

## Archive complete diagnostic

Once all 12 reports and \`analysis_summary.json\` exist:

\`\`\`bash
git add -f reports/train_surface_cross_eval_v0/seed*_cross_eval.json \
  reports/train_surface_cross_eval_v0/analysis_summary.json
git commit -m "analysis: archive all read-only cross-surface seed audits"
git push
\`\`\`

**The summary is exploratory:** the source choices and question were motivated by already-seen bare-only results. A new, independently trained cohort would be needed for a confirmatory test of the observed surface-rescue pattern, if any.
