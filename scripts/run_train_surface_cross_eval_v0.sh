#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 14901 14903 14907 14909 14913 14919 14921 14927 14931 14933 14939 14943; do
  report=$(printf 'reports/train_surface_cross_eval_v0/seed%d_cross_eval.json' "$seed")
  if [[ -e "$report" ]]; then echo "SKIP sealed read-only diagnostic $seed"; continue; fi
  python3 -u scripts/train_surface_cross_eval_v0.py --phase score --device mps --seed "$seed"
done
python3 scripts/train_surface_cross_eval_v0.py --phase summary
