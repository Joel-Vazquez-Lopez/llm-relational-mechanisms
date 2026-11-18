#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
script=scripts/phase2_02_structural_transfer_v0.py
python3 "$script" --phase prepare
for seed in 11801 11803 11807 11809 11813 11819 11821 11827; do
  for repeat in 1 2; do
    report="reports/phase2_02/structural_transfer_v0/seed${seed}_repeat${repeat}_mps_step6400.json"
    checkpoint="results/phase2_02/structural_transfer_v0/seed${seed}_repeat${repeat}_mps_step6400.pt"
    if [[ -f "$report" && -f "$checkpoint" ]]; then
      echo "SKIP sealed report + checkpoint: $seed / $repeat"
      continue
    fi
    if [[ -f "$report" || -f "$checkpoint" ]]; then
      echo "ERROR partial report/checkpoint pair; do not overwrite. Inspect $seed / $repeat" >&2
      exit 1
    fi
    echo "===== PHASE2-02 training seed=$seed repeat=$repeat ====="
    python3 -u "$script" --phase run --seed "$seed" --repeat "$repeat" --device mps
  done
done
python3 "$script" --phase summary
