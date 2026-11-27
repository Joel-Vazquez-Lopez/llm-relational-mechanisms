#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 16101 16103 16107 16109 16113 16119 16121 16127 16131 16133 16139 16143 16149 16157 16161 16167; do
  report=$(printf 'reports/role_diversity_complex_v0/seed%d_mps_step6400.json' "$seed")
  weights=$(printf 'results/role_diversity_complex_v0/seed%d_mps_step6400.pt' "$seed")
  if [[ -f "$report" && -f "$weights" ]]; then echo "SKIP SEALED $seed"; continue; fi
  if [[ -f "$report" || -f "$weights" ]]; then echo "PARTIAL SEED $seed — STOP, never overwrite" >&2; exit 2; fi
  python3 -u scripts/role_diversity_complex_v0.py --phase run --seed "$seed" --device mps
done
python3 scripts/role_diversity_complex_v0.py --phase summary
