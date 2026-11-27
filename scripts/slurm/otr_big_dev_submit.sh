#!/bin/bash
# Dev pass for the two 70B-class models (Llama-3.3-70B-Instruct, Qwen2.5-72B-Instruct), each split over
# 2 x H100 with --device-map auto. Same dev measurements the core five models were frozen from
# (FREEZE.md section 4): behaviour (C1), role-layer profile (OTR-03, ordinary and failing question),
# residual curve + head scan (OTR-09, with all controls), "by" layer (OTR-13), neutral prefix (OTR-07).
# OTR-14 needs the dev OTR-09 file (--rank-from) and is submitted afterwards (wave 2).
# Run the smoke job (otr_big_smoke.sbatch) first and only continue if it prints SMOKE PASSED.
#     bash scripts/slurm/otr_big_dev_submit.sh            # prints the commands only (dry run)
#     bash scripts/slurm/otr_big_dev_submit.sh --submit   # submits them
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ACCT="uppmax2026-1-225"
SUBMIT=0; [[ "${1:-}" == "--submit" ]] && SUBMIT=1
INSTR="Reply with one name only."
FAIL='Who was someone {past} by?'
WORK='Who was the {ing} done by?'
MODELS=("meta-llama/Llama-3.3-70B-Instruct" "Qwen/Qwen2.5-72B-Instruct")

run() {  # $1 = label, $2 = time limit, rest = sbatch arguments
  local label="$1" t="$2"; shift 2
  printf '%s:' "$label"; printf ' %q' sbatch -A "$ACCT" --gpus=h100:2 -t "$t" "$@"; echo
  if [[ $SUBMIT == 1 ]]; then
    local out; out=$(sbatch -A "$ACCT" --gpus=h100:2 -t "$t" "$@")
    echo "  $out"
  fi
}

# Order matters a little: OTR-09 first (heads and handover; OTR-14 waits for it), then OTR-03.
for M in "${MODELS[@]}"; do
  C=(--chat-template --split dev --device-map auto)
  run "OTR-09 dev $M"   16:00:00 scripts/slurm/otr09_pelle.sbatch "$M" "${C[@]}" --instruction "$INSTR" --controls --controls-extra --n-random 5
done
for M in "${MODELS[@]}"; do
  C=(--chat-template --split dev --device-map auto)
  run "OTR-03 dev $M"   16:00:00 scripts/slurm/otr03_pelle.sbatch "$M" "${C[@]}" --layer-stride 2
  run "OTR-03 fail $M"  16:00:00 scripts/slurm/otr03_pelle.sbatch "$M" "${C[@]}" --layer-stride 2 --roles agent --agent-question "$FAIL" --include-incorrect
  run "OTR-13 dev $M"   12:00:00 scripts/slurm/otr13_pelle.sbatch "$M" "${C[@]}" --instruction "$INSTR"
  run "C1 fail $M"      06:00:00 scripts/slurm/otrgen_pelle.sbatch "$M" "${C[@]}" --n 512 --instruction "$INSTR" --question-template "$FAIL" --question-role agent
  run "C1 work $M"      06:00:00 scripts/slurm/otrgen_pelle.sbatch "$M" "${C[@]}" --n 512 --instruction "$INSTR" --question-template "$WORK" --question-role agent
  run "OTR-07 dev $M"   08:00:00 scripts/slurm/otr07_pelle.sbatch "$M" "${C[@]}" --instruction "$INSTR" --by-prefix "Someone was {verb} by" --condition agent "$FAIL" --condition agent "$WORK"
done
[[ $SUBMIT == 1 ]] && echo "Submitted. Wave 2 (OTR-14) once the OTR-09 dev files exist." || echo "(dry run: nothing submitted)"
