# ROLE-ANCHOR-PROBE-v0 — Mac run guide

Use the original 12-seed ×4-arm model checkpoints; **no new training**.

\`\`\`bash
git status --short
git fetch origin
git switch --track origin/analysis/role-anchor-probe-v0
python3 scripts/train_role_anchor_probe_v0.py --phase smoke --device mps
\`\`\`

Smoke creates a disposable random model and tests no original stored checkpoints.

With original checkpoint files at \`results/train_surface_factorial_v0/seed14901_mps_step6400.pt\` and the corresponding other 149xx names:

\`\`\`bash
caffeinate -dimsu bash scripts/run_train_role_anchor_probe_v0.sh
\`\`\`

This audits all 48 models, writes one immutable per-seed JSON result and refuses an incomplete final aggregate. It loads checkpoints strictly read-only and verifies that the role-anchor script and its protocol are committed in Git before scoring.

When all 12 reports and \`analysis_summary.json\` exist:

\`\`\`bash
git add -f reports/train_role_anchor_probe_v0/seed*_role_anchor.json reports/train_role_anchor_probe_v0/analysis_summary.json
git commit -m "analysis: archive all read-only original-role anchor probes"
git push -u origin analysis/role-anchor-probe-v0
\`\`\`

Do not push .pt checkpoint files or unrelated working directories. The results are exploratory, not an independent training replication.
