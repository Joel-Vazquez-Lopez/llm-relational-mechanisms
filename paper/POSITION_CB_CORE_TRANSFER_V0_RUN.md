# Run CORE-TRANSFER-v0 on your existing Mac checkpoints

The next audit is **read-only** and uses the same local 32 final POSITION-CB checkpoints. No new neural training, no protected evaluation.

From your existing research checkout, with the same PyTorch/MPS environment as the previous audit:

    git status --short
    git fetch origin
    git switch --track origin/analysis/position-cb-core-transfer-v0
    python3 scripts/position_cb_core_transfer_v0.py --phase prepare
    python3 scripts/position_cb_core_transfer_v0.py --phase smoke --device mps

If the branch already exists locally, switch to it without --track. Do not discard unrelated work. If either preflight fails, stop and share the traceback.

Before any checkpoint inference, commit the generated manifest:

    git add reports/position_cb_core_transfer_v0/manifest.json
    git commit -m "research: freeze core-transfer sources before inference"
    git push -u origin analysis/position-cb-core-transfer-v0

Then launch all original seed/repeat pairs:

    caffeinate -dimsu bash scripts/run_position_cb_core_transfer_v0.sh

The full archive lands in reports/position_cb_core_transfer_v0/analysis_summary.json. To share it, commit only JSON:

    git add -f reports/position_cb_core_transfer_v0/*.json
    git commit -m "analysis: complete core-transfer read-only audit"
    git push origin analysis/position-cb-core-transfer-v0

Never commit .pt checkpoints. Results are exploratory, and ALL original seeds remain included.