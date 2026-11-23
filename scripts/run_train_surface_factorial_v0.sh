#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 14901 14903 14907 14909 14913 14919 14921 14927 14931 14933 14939 14943; do
  report=$(printf 'reports/train_surface_factorial_v0/seed%d_mps_step6400.json' "$seed")
  weights=$(printf 'results/train_surface_factorial_v0/seed%d_mps_step6400.pt' "$seed")
  if [[ -f "$report" && -f "$weights" ]]; then echo "SKIP SEALED $seed";continue;fi
  if [[ -f "$report" || -f "$weights" ]]; then echo "PARTIAL seed $seed; STOP" >&2;exit 2;fi
  python3 -u scripts/train_surface_factorial_v0.py --phase run --seed "$seed" --device mps
done
python3 scripts/train_surface_factorial_v0.py --phase summary
