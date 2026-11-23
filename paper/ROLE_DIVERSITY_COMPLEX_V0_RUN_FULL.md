# ROLE-DIVERSITY-COMPLEX-v0 — Mac execution guide

A **NEW 16-seed ×4-arm =64 model scratch-training study**, not the read-only old checkpoint probe. Do not execute scientific runs before both manifests are frozen in Git. Preserve old 149xx studies and all unrelated local folders.

## 1. Get branch

\`\`\`bash
git status --short
git fetch origin
git switch --track origin/design/role-diversity-complex-v0
\`\`\`

If branch already exists use git switch design/role-diversity-complex-v0; do not force-switch past uncommitted changes. Use the same Python/PyTorch environment that handled the earlier MPS training.

## 2. Source-only checks and disposable training-only smoke

\`\`\`bash
python3 scripts/role_diversity_complex_v0_preflight.py --phase prepare
python3 scripts/role_diversity_complex_v0.py --phase prepare
python3 scripts/role_diversity_complex_v0.py --phase smoke --device mps
\`\`\`

First command audits the nested 1/2/4/8 role-source sets; second audits full per-seed schedules and creates a prospective source/study manifest, with no model inference; third runs only one disposable update per arm on current TRAIN-eligible examples, never unseen-role tests. If any check fails, STOP.

## 3. Freeze before first prospective training

\`\`\`bash
git add -f \
  reports/role_diversity_complex_v0/design_source_audit.json \
  reports/role_diversity_complex_v0/study_manifest.json

git commit -m "research: freeze ROLE-DIVERSITY-COMPLEX v0"
git push -u origin design/role-diversity-complex-v0
\`\`\`

The training script refuses any changed or uncommitted scientific input and checks all schedule and protocol hashes against the manifests.

## 4. Run all 16 fresh seeds

\`\`\`bash
caffeinate -dimsu bash scripts/run_role_diversity_complex_v0.sh
\`\`\`

Each seed trains four matched scratch models. No optional stopping or cherry-picking. If interrupted, launcher skips complete sealed checkpoint+JSON pairs but stops on a partial pair. Never erase a partial checkpoint to rewrite a result.

## 5. Archive ONLY when all 16 complete

\`\`\`bash
git add -f \
  reports/role_diversity_complex_v0/seed*_mps_step6400.json \
  reports/role_diversity_complex_v0/analysis_summary.json
git commit -m "study: archive complete ROLE-DIVERSITY-COMPLEX v0 results"
git push
\`\`\`

Do not commit .pt weights, order-role-patches or unrelated files.

**Frozen primary:** does training with 8 distinct role assignments, rather than 1, improve rich heldout-role joint accuracy by at least 10 pp with seed-level positive CI and at least 12/16 favorable seeds? **Only if primary passes**, check whether 2 distinct repeatedly presented mappings already yield +10 pp with positive CI. This is not a one-shot exposure study.
