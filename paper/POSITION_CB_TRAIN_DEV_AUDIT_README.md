# POSITION-CB training-vs-development audit — run from local checkpoints

**What is finished:** original 8-seed position-counterbalanced experiment (16 processes / 32 trained models), frozen manifest and primary/secondary results. **What this is:** a cheap *post-hoc diagnostic*, not a new confirmatory experiment. It does not train any model or open protected/reserved/L+I examples.

## Why

Several independently trained models have very low familiar-development correctness, yet their final *training minibatch* losses are nearly zero. A low loss on a sampled minibatch cannot distinguish memorization from distributional generalization. Before launching a second full training study, contrast those outcomes using **the existing final local checkpoints**.

This script evaluates for each training arm:

1. **Exact training-presented source variants:** 144 deterministic pool indices selected by the model's 6,400-step schedule (each evaluated on the two voice-combinations actually used in training)
2. **Never-selected pool variants:** up to 144 indices from the *same train-only source pool* that this seed's schedule did not select even once (they may share core names and events with observed examples)
3. **Familiar development performance:** the *existing archived score* from the original run JSON; no fresh development predictions

The comparison is intentionally descriptive and holds out no new syntax family. Its scientific value is to distinguish successful fitting of seen items from transfer to other items sharing the vocabulary/task distribution.

## Run it on the Mac with the existing checkpoints

On your original checkout, verify local work before switching branches:

\`\`\`bash
git status --short
git fetch origin
git switch --track origin/analysis/position-cb-train-dev-audit
source .venv/bin/activate
python3 scripts/position_cb_train_dev_audit_v0.py --phase preflight
\`\`\`

If you are already on the named branch, use \`git switch analysis/position-cb-train-dev-audit\`. If your virtual environment lives elsewhere, activate it as before. Local \`results/position_cb_v0/*.pt\` files must remain present at their original paths (these files were *never* in GitHub).

Then run:

\`\`\`bash
caffeinate -dimsu bash scripts/run_position_cb_train_dev_audit_v0.sh
\`\`\`

The script writes small JSON records under \`reports/position_cb_train_dev_audit_v0/\`. No checkpoints are overwritten, no optimizer steps are performed, and no item-level answers or texts are written. The launcher can be rerun safely and skips already complete diagnostic records.

Inspect \`reports/position_cb_train_dev_audit_v0/analysis_summary.json\` only after all 16 runs are complete. Commit the JSON report files and share the updated branch with me for the paper, but **never** commit \`.pt\` checkpoints.

## Decision after the diagnostic

- High performance on exact train variants, low performance on train-pool never-selected variants and low dev: evidence of memorization/poor systematic transfer; prioritize lexical/source-core augmentation or a train-distribution redesign **chosen using training-only/familiar validation**, then freeze a new study.
- High on both train-pool groups, low dev: stronger evidence for a partition/domain shift (source-core/name-set, lexical identity/context differences); identify the shift before new-grammar training.
- Low on exact train variants despite low final sampled minibatch loss: test whether the last-batch loss misrepresents training-set competence, interference/forgetting, or a mismatch between optimizer's training samples and evaluation.
- Strong on all three: the instability is seed/model-specific; evaluate whether a fresh-seed replication at a stable training regime is warranted.

Do **not** change the frozen primary study outcome, exclude failed seeds, or retroactively reinterpret the original nominally passing confidence intervals.
