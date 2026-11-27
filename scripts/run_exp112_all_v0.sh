#!/usr/bin/env bash
# EXP112 fixed exploratory 48-model full analysis; no checkpoint modifications.
set -euo pipefail
cd "$(dirname "$0")/.."
script=scripts/exp112_clause_state_causal_patch_v0.py
python3 "$script" --phase prepare
for seed in 10901 10903 10907 10909 10913 10919 10921 10927; do
  for repeat in 1 2; do
    for arm in pa_only indirect_pp_aa direct_pa_ap; do
      result="reports/exp112/clause_state_patch_v0/seed${seed}_repeat${repeat}_${arm}.json"
      if [[ -e "$result" ]]; then
        echo "SKIP existing result: $result"
        continue
      fi
      echo "===== EXP112 seed=$seed repeat=$repeat arm=$arm ====="
      python3 -u "$script" --phase run --seed "$seed" --repeat "$repeat" --arm "$arm" --device mps
    done
  done
done
python3 "$script" --phase summary
echo 'EXP112 COMPLETED: 48/48 model diagnostics'
