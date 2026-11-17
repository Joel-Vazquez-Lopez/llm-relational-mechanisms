#!/usr/bin/env bash
# EXP-113 sequential read-only analysis. Start from experimental worktree root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p reports/exp113/position_update_factorial_v0
for seed in 10901 10903 10907 10909 10913 10919 10921 10927; do
  for repeat in 1 2; do
    for arm in pa_only indirect_pp_aa direct_pa_ap; do
      path="reports/exp113/position_update_factorial_v0/seed${seed}_repeat${repeat}_${arm}.json"
      if [[ -f "$path" ]]; then
        echo "SKIP $path"
        continue
      fi
      echo "===== EXP113 seed=$seed repeat=$repeat arm=$arm ====="
      python3 -u scripts/exp113_position_update_factorial_v0.py \
        --phase run --seed "$seed" --repeat "$repeat" --arm "$arm" --device mps
    done
  done
done
python3 scripts/exp113_position_update_factorial_v0.py --phase summary
