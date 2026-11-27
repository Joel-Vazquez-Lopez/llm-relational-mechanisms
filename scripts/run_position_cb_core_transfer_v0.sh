#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
script=scripts/position_cb_core_transfer_v0.py
python3 "$script" --phase prepare
for seed in 12701 12703 12707 12709 12713 12719 12721 12727; do
  for repeat in 1 2; do
    report=$(printf 'reports/position_cb_core_transfer_v0/seed%s_repeat%s_core_audit.json' "$seed" "$repeat")
    if [[ -f "$report" ]]; then
      echo "SKIP sealed report $seed / $repeat"
      continue
    fi
    echo "===== CORE TRANSFER seed $seed repeat $repeat ====="
    python3 -u "$script" --phase run --seed "$seed" --repeat "$repeat" --device mps
  done
done
python3 "$script" --phase summary
