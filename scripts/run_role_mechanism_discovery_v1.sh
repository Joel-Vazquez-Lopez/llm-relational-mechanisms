#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -u scripts/role_mechanism_discovery_v1.py --phase run-all
python -u scripts/role_mechanism_discovery_v1.py --phase summary
