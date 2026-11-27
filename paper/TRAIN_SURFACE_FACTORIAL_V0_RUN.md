# TRAIN-SURFACE-FACTORIAL-v0 — Apple MPS run guide

**Study status:** implemented; source-only and CPU throwaway-model checks run in GitHub Actions; **no scientific models have yet been trained.** The source/study manifests and complete runner must be frozen in Git before starting any 149xx seed.

The completed CORE-DIVERSITY-v1 and position-counterbalanced experiments are **not** part of this execution; never delete, overwrite or retrain them. Preserve unrelated untracked local directories (including \`order-role-patches/\`). Do not stage those folders here.

## 1. Get the experiment branch

From the root of the original repo, with its working Python 3 environment activated:

\`\`\`bash
git status --short
git fetch origin
git switch --track origin/experiments/train-surface-factorial-v0
\`\`\`

If the branch already exists locally, use \`git switch experiments/train-surface-factorial-v0\` instead. Do not force a switch across overlapping uncommitted edits; leave unrelated work alone.

## 2. Prepare and verify, with NO prospective model inference

\`\`\`bash
python3 scripts/train_surface_factorial_v0_preflight.py --phase prepare
python3 scripts/train_surface_factorial_v0.py --phase prepare
python3 scripts/train_surface_factorial_v0.py --phase smoke --device mps
\`\`\`

The source preflight verifies 21,600 TRAIN surface variants and exactly **7,680 unseen-role** and **2,880 withheld-quartet** source presentations, independent gold parsing, train/evaluation disjointness, 48-token input ceiling, 45 balanced nuisance-feature cells and the unchanged historical 240/60 TRAIN-quartet split. It also reports the intentionally *unmatched* length/position/name-mention marginals of the four treatment distributions.

The manifest preparation hashes all 12 fresh seed schedules and complete experiment code/dependencies. The smoke initializes disposable weights, checks four on-TRAIN-source mini-optimizer updates and native scorer categories. **It never scores heldout role test items or reserved/protected inputs.**

## 3. Freeze BOTH manifests before any scientific run

Only after all three checks pass:

\`\`\`bash
git add -f \
  reports/train_surface_factorial_v0/source_manifest.json \
  reports/train_surface_factorial_v0/study_manifest.json
git commit -m "research: freeze TRAIN-SURFACE-FACTORIAL v0 sources and schedules"
git push -u origin experiments/train-surface-factorial-v0
\`\`\`

The scientific runner verifies that the runner, preflight, launcher, protocol, and **both manifests match committed bytes**. Any drift blocks training. If a check fails, stop rather than trying to modify a frozen experiment after inference.

## 4. Execute once, then analyze only after all 12 seeds

\`\`\`bash
caffeinate -dimsu bash scripts/run_train_surface_factorial_v0.sh
\`\`\`

Fixed design: 12 NEW independent seeds × 4 training arms = 48 models; 6,400 optimizer updates per model. Every treatment sees the same eight training quartets and label assignments per step, differing only by neutral prefixes and/or named adjunct conditions.

The launcher skips a seed only when **both** its sealed JSON report and local checkpoint exist. If exactly one exists, it STOPS to prevent overwriting; do not bypass this guard. Scientific runs use Apple MPS. Keep .pt checkpoints locally; only upload the results JSON after the **complete** 12-seed summary.

**Do not inspect interim heldout predictions, choose favorite seeds, change the number of steps or revise the primary effect/threshold.** Primary requires a seed-mean adjunct-associated reduction >=10 pp, seed-level CI lower bound >0 and >=9/12 seeds positive. A failed primary closes the gated withheld-quartet test.

## 5. Archive the full cohort

Only after 12 new JSON seed reports and \`analysis_summary.json\` exist:

\`\`\`bash
git add -f \
  reports/train_surface_factorial_v0/seed*_mps_step6400.json \
  reports/train_surface_factorial_v0/analysis_summary.json
git commit -m "study: archive complete TRAIN-SURFACE-FACTORIAL v0 results"
git push
\`\`\`

Then review all seed-level effects, original-core competence, common heldout role/per-quartet results, and paired-answer classes together. A secondary effect is not confirmatory if the primary gate fails.

## Results interpretation

This is a *new* prospectively specified single-grammar causal study built after observing the earlier full bare/bare results. Its design changes **training source lengths, name mentions and token positions** by construction. A positive result would show a within-grammar treatment-distribution effect, not isolate a unique head or proof of a semantic-role circuit. A null result remains an important result and must not trigger iterative threshold changes or selective reruns.
