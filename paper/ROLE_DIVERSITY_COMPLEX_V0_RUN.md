# ROLE-DIVERSITY-COMPLEX-v0 — source-only design (NOT ready for training)

This is a **design branch only**. No new scientific training runner or frozen 16-seed study manifest exists yet. Do not launch new MPS training from this branch.

From your repository:

\`\`\`bash
git fetch origin
git switch --track origin/design/role-diversity-complex-v0
python3 scripts/role_diversity_complex_v0_preflight.py --phase check
# Optional immutable local design manifest:
python3 scripts/role_diversity_complex_v0_preflight.py --phase prepare
\`\`\`

The script renders and checks source-only fixtures; it does NOT load checkpoints or perform inference and never uses original DEV, reserved, protected or L+I materials.

Design: nested training-core diversity 1 / 2 / 4 / 8, same complex prefix+named-adjunct grammar, same 240 old TRAIN quartets, common withheld 16 role permutations and 60 withheld TRAIN-derived quartets. Separate role count from literal *number of repeated exposures*: the \`roles_2\` arm intends to show one additional combination across multiple updates, NOT once.

Full next steps before any training: frozen schedule and role × surface-cell exposure audit, exact model runner and scoring, fresh seed list, closed primary and secondary gates, paired initialization tests, source/study manifests committed before first inference. Until then complete the **read-only ROLE-ANCHOR-PROBE-v0** existing-checkpoint investigation.
