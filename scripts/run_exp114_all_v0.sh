#!/usr/bin/env bash
# EXP-114 fixed 16 training-run launcher. Requires a committed frozen manifest.
# Runs all 3 arms per seed/repeat; complete 48 trained model states.
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 11401 11403 11407 11409 11413 11419 11421 11427; do
  for repeat in 1 2; do
    report="reports/exp114/independent_replication_v0/seed${seed}_repeat${repeat}_mps_step6400.json"
    checkpoint="results/exp114/independent_replication_v0/seed${seed}_repeat${repeat}_mps_step6400.pt"
    if [[ -f "$report" && -f "$checkpoint" ]]; then
      echo "SKIP already completed: $seed / $repeat"
      continue
    fi
    if [[ -f "$report" || -f "$checkpoint" ]]; then
      echo "PARTIAL RUN DETECTED for $seed/$repeat. Stop and inspect; never overwrite." >&2
      exit 1
    fi
    echo "===== TRAIN AND SEALED INTERVENTIONS: SEED $seed REP $repeat ====="
    python3 -u scripts/exp114_independent_replication_v0.py \
      --phase run --seed "$seed" --repeat "$repeat" --device mps
  done
done
printf '\n===== ALL 16 TRAINING RUNS COMPLETE; FIXED SUMMARY =====\n'
python3 scripts/exp114_independent_replication_v0.py --phase summary
