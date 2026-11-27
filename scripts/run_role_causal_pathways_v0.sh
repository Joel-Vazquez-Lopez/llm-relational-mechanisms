#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-mps}"
for seed in 16101 16127 16143 16167; do
  for step in 6400 12800 25600 51200; do
    "$PYTHON" -u scripts/role_causal_pathways_v0.py --phase run --seed "$seed" --step "$step" --device "$DEVICE"
  done
done
"$PYTHON" scripts/role_causal_pathways_v0.py --phase summary
