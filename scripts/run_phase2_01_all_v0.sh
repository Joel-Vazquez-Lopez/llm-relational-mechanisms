#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
script=scripts/phase2_01_coverage_curve_v0.py
python3 "$script" --phase prepare
for seed in 11701 11703 11707 11709 11713 11719 11721 11727; do
  for repeat in 1 2; do
    report="reports/phase2_01/coverage_curve_v0/seed${seed}_repeat${repeat}_mps_step6400.json"
    checkpoint="results/phase2_01/coverage_curve_v0/seed${seed}_repeat${repeat}_mps_step6400.pt"
    if [[ -e "$report" && -e "$checkpoint" ]]; then
      echo "SKIP existing complete pair seed=$seed repeat=$repeat"
      continue
    fi
    if [[ -e "$report" || -e "$checkpoint" ]]; then
      echo "ERROR: partial pair exists; do not overwrite $report $checkpoint" >&2
      exit 1
    fi
    echo "===== PHASE2-01 seed=$seed repeat=$repeat (five arms) ====="
    python3 -u "$script" --phase run --seed "$seed" --repeat "$repeat" --device mps
  done
done
python3 "$script" --phase summary
