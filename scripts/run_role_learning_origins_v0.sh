#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(git branch --show-current)" == experiments/role-learning-origins-v0 ]]
python scripts/role_learning_origins_v0.py --phase run
python scripts/role_learning_origins_v0.py --phase summary
