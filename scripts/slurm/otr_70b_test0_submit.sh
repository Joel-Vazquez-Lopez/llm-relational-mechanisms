#!/bin/bash
# Submit the frozen 70B test0 commands (docs/order_to_role/FREEZE_70B.md). Run ONCE, on a Pelle login node, from the
# repository root of a checkout at the commit that adds FREEZE_70B.md:
#     bash scripts/slurm/otr_70b_test0_submit.sh            # prints the commands only (dry run)
#     bash scripts/slurm/otr_70b_test0_submit.sh --submit   # submits them and logs the job IDs
# 10 jobs per model (C1 x2, C2, C3, C4/C5, C6/C8, C7; BX C2, C3, C5) = 20 jobs, each on 2 x H100 (--device-map auto),
# 4-hour limit (dev runs of the same size took 0.5-2.1 h). Job IDs go to docs/order_to_role/test0_attempts.md.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ACCT="uppmax2026-1-225"
SUBMIT=0; [[ "${1:-}" == "--submit" ]] && SUBMIT=1
LOG=docs/order_to_role/test0_attempts.md
COMMIT=$(git rev-parse --short HEAD)
INSTR="Reply with one name only."
FAIL='Who was someone {past} by?'
WORK='Who was the {ing} done by?'

for f in docs/order_to_role/FREEZE_70B.md scripts/otr_freeze_score.py; do
  [[ -f "$f" ]] || { echo "missing $f"; exit 1; }
  git diff --quiet HEAD -- "$f" || { echo "$f has uncommitted changes; commit the freeze first"; exit 1; }
done

# model | role layer | "by" layer | handover | dev OTR-09 file (frozen heads)
MODELS=(
  "meta-llama/Llama-3.3-70B-Instruct|34|14|38|otr09_meta-llama__Llama-3.3-70B-Instruct_dev_20261127T121453.json"
  "Qwen/Qwen2.5-72B-Instruct|48|30|58|otr09_Qwen__Qwen2.5-72B-Instruct_dev_20261127T121538.json"
)

run() {  # $1 = label, rest = sbatch arguments
  local label="$1"; shift
  printf '%s:' "$label"; printf ' %q' sbatch -A "$ACCT" --gpus=h100:2 -t 04:00:00 "$@"; echo
  if [[ $SUBMIT == 1 ]]; then
    local out; out=$(sbatch -A "$ACCT" --gpus=h100:2 -t 04:00:00 "$@")
    echo "  $out"
    echo "| $(date '+%Y-%m-%d %H:%M') | $COMMIT | 70B $label | ${out##* } | submitted |" >> "$LOG"
  fi
}

even_upto() { local s=""; for ((l=0; l<=$1; l+=2)); do s+="${s:+,}$l"; done; echo "$s"; }

for row in "${MODELS[@]}"; do
  IFS='|' read -r M ROLE BY HAND RF <<< "$row"
  RFP="results/$RF"
  [[ -f "$RFP" ]] || { echo "missing $RFP"; exit 1; }
  GRID09=$(even_upto $((HAND + 2)))
  C=(--chat-template --split test0 --device-map auto)
  run "C4/C5 $M"    scripts/slurm/otr09_pelle.sbatch "$M" "${C[@]}" --instruction "$INSTR" --rank-from "$RFP" --skip-scan --n-random 5 --controls --controls-extra --layers "$GRID09"
  run "C1 fail $M"  scripts/slurm/otrgen_pelle.sbatch "$M" "${C[@]}" --n 512 --instruction "$INSTR" --question-template "$FAIL" --question-role agent
  run "C1 work $M"  scripts/slurm/otrgen_pelle.sbatch "$M" "${C[@]}" --n 512 --instruction "$INSTR" --question-template "$WORK" --question-role agent
  run "C2 $M"       scripts/slurm/otr03_pelle.sbatch "$M" "${C[@]}" --n-events 64 --layers "$ROLE"
  run "C3 $M"       scripts/slurm/otr03_pelle.sbatch "$M" "${C[@]}" --n-events 64 --layers "$ROLE" --roles agent --agent-question "$FAIL" --include-incorrect
  run "C6/C8 $M"    scripts/slurm/otr14_pelle.sbatch "$M" "${C[@]}" --rank-from "$RFP" --layers "$BY"
  run "C7 $M"       scripts/slurm/otr07_pelle.sbatch "$M" "${C[@]}" --instruction "$INSTR" --by-prefix "Someone was {verb} by" --condition agent "$FAIL" --condition agent "$WORK"
  B=(--chat-template --split test0 --device-map auto --stimuli bx-transitive)
  run "BX C2 $M"    scripts/slurm/otr03_pelle.sbatch "$M" "${B[@]}" --n-events 48 --layers "$ROLE"
  run "BX C3 $M"    scripts/slurm/otr03_pelle.sbatch "$M" "${B[@]}" --n-events 48 --layers "$ROLE" --agent-question "$FAIL" --include-incorrect
  run "BX C5 $M"    scripts/slurm/otr09_pelle.sbatch "$M" "${B[@]}" --instruction "$INSTR" --rank-from "$RFP" --skip-scan --n-random 5 --layers "$GRID09"
done
[[ $SUBMIT == 1 ]] && echo "Logged to $LOG." || echo "(dry run: nothing submitted)"
