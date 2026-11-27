#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(git branch --show-current)" != "experiments/role-mechanism-repair-v0" ]]; then
  echo 'Wrong branch: switch to experiments/role-mechanism-repair-v0 before running.' >&2
  exit 1
fi
python -u scripts/role_mechanism_repair_v0.py --phase run-all
python -u scripts/role_mechanism_repair_v0.py --phase summary
