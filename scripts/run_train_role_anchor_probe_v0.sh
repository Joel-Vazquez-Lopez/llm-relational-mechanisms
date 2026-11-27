#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 14901 14903 14907 14909 14913 14919 14921 14927 14931 14933 14939 14943; do
  report=$(printf 'reports/train_role_anchor_probe_v0/seed%d_role_anchor.json' "$seed")
  if [[ -f "$report" ]]; then echo "SKIP sealed role-anchor probe $seed"; continue; fi
  python3 -u scripts/train_role_anchor_probe_v0.py --phase run --seed "$seed" --device mps
done
python3 scripts/train_role_anchor_probe_v0.py --phase summary
