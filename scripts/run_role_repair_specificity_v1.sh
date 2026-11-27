#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $(git branch --show-current) == experiments/role-repair-specificity-v1 ]] || { echo 'Wrong branch'; exit 1; }
python scripts/role_repair_specificity_v1.py --phase run-all
python scripts/role_repair_specificity_v1.py --phase summary
