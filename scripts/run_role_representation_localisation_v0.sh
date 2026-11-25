#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 16101 16127 16143 16167; do
  for step in 6400 12800 25600 51200; do
    python -u scripts/role_representation_localisation_v0.py --phase run --seed "$seed" --step "$step"
  done
done
python scripts/role_representation_localisation_v0.py --phase summary
