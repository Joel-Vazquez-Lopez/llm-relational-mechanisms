#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in $(seq 27101 27116); do
  python -u scripts/role_mechanism_replication_v0.py --phase run --seed "$seed"
done
python -u scripts/role_mechanism_replication_v0.py --phase summary
