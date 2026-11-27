#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
script=scripts/position_counterbalance_v0.py
python3 "$script" --phase prepare
for seed in 12701 12703 12707 12709 12713 12719 12721 12727; do
  for repeat in 1 2; do
    report="reports/position_cb_v0/seed${seed}_repeat${repeat}_mps_step6400.json"
    checkpoint="results/position_cb_v0/seed${seed}_repeat${repeat}_mps_step6400.pt"
    if [[ -f "$report" && -f "$checkpoint" ]]; then
      echo "SKIP completed pair $seed / $repeat"
      continue
    fi
    if [[ -f "$report" || -f "$checkpoint" ]]; then
      echo "ERROR: incomplete pair $seed / $repeat. Never overwrite a sealed result." >&2
      exit 1
    fi
    echo "===== POSITION-CB seed=$seed repeat=$repeat ====="
    python3 -u "$script" --phase run --seed "$seed" --repeat "$repeat" --device mps
  done
done
python3 "$script" --phase summary
