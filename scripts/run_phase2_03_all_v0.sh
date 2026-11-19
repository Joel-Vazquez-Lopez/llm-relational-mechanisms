#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source_script=scripts/phase2_03_distractor_event_binding_v0.py
python3 "$source_script" --phase prepare
for seed in 11901 11903 11907 11909 11913 11919 11921 11927; do
  for repeat in 1 2; do
    report="reports/phase2_03/distractor_event_binding_v0/seed${seed}_repeat${repeat}_mps_step6400.json"
    checkpoint="results/phase2_03/distractor_event_binding_v0/seed${seed}_repeat${repeat}_mps_step6400.pt"
    if [[ -f "$report" && -f "$checkpoint" ]]; then
      echo "SKIP completed report and checkpoint: $report"
      continue
    fi
    if [[ -f "$report" || -f "$checkpoint" ]]; then
      echo "ERROR: partial run found for $seed repeat $repeat; refusing to skip" >&2
      exit 1
    fi
    echo "===== PHASE2-03 seed=$seed repeat=$repeat ====="
    python3 -u "$source_script" --phase run --seed "$seed" --repeat "$repeat" --device mps
  done
done
python3 "$source_script" --phase summary
