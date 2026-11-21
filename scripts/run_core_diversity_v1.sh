#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 13901 13903 13907 13909 13913 13919 13921 13927 13931 13933 13939 13943 13949 13951 13959 13963; do
  report=$(printf 'reports/core_diversity_v1/seed%d_mps_step6400.json' "$seed")
  checkpoint=$(printf 'results/core_diversity_v1/seed%d_mps_step6400.pt' "$seed")
  if [[ -f "$report" && -f "$checkpoint" ]]; then
    echo "SKIP existing sealed seed $seed"
    continue
  fi
  if [[ -f "$report" || -f "$checkpoint" ]]; then
    echo "PARTIAL RUN EXISTS seed $seed: stop, do not overwrite" >&2
    exit 1
  fi
  echo "===== CORE-DIVERSITY-v1 seed $seed ====="
  python3 -u scripts/core_diversity_v1.py --phase run --seed "$seed" --device mps
done
python3 scripts/core_diversity_v1.py --phase summary
