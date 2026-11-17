#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
script=scripts/exp115_block1_head_mlp_causal_v0.py
python3 "$script" --phase prepare
for seed in 11401 11403 11407 11409 11413 11419 11421 11427; do
  for repeat in 1 2; do
    for arm in pa_only indirect_pp_aa direct_pa_ap; do
      result="reports/exp115/block1_head_mlp_causal_v0/seed${seed}_repeat${repeat}_${arm}.json"
      if [[ -e "$result" ]]; then
        echo "SKIP existing report $result"
        continue
      fi
      echo "===== EXP115 seed=$seed repeat=$repeat arm=$arm ====="
      python3 -u "$script" --phase run --seed "$seed" --repeat "$repeat" --arm "$arm" --device mps
    done
  done
done
python3 "$script" --phase summary
