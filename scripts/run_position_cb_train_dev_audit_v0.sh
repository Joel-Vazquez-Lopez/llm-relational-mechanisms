#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/position_cb_train_dev_audit_v0.py --phase preflight
for seed in 12701 12703 12707 12709 12713 12719 12721 12727; do
  for repeat in 1 2; do
    echo "===== read-only checkpoint audit $seed / $repeat ====="
    python3 -u scripts/position_cb_train_dev_audit_v0.py \
      --phase run --seed "$seed" --repeat "$repeat" --device mps
  done
done
python3 scripts/position_cb_train_dev_audit_v0.py --phase summary
