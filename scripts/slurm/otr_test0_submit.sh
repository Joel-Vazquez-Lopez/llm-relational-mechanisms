#!/bin/bash
# Submit the frozen test0 batch (docs/order_to_role/FREEZE.md, section 5). Run ONCE, on a Pelle login node,
# from the repository root, at the freeze commit:
#     bash scripts/slurm/otr_test0_submit.sh            # prints the commands only (dry run)
#     bash scripts/slurm/otr_test0_submit.sh --submit   # submits them and logs the job IDs
# Every job runs on one H100 (as frozen). Job IDs are appended to docs/order_to_role/test0_attempts.md.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ACCT="uppmax2026-1-225"
SUBMIT=0; [[ "${1:-}" == "--submit" ]] && SUBMIT=1
LOG=docs/order_to_role/test0_attempts.md
COMMIT=$(git rev-parse --short HEAD)
INSTR="Reply with one name only."
FAIL='Who was someone {past} by?'
WORK='Who was the {ing} done by?'

# model | extra flag | role layer | "by" layer | handover layer | dev OTR-09 file | C7 claim
MODELS=(
  "google/gemma-4-31B-it||28|28|40|otr09_google__gemma-4-31B-it_dev_20261118T164326.json|1"
  "Qwen/Qwen3.8-27B|--no-thinking|29|30|54|otr09_Qwen__Qwen3.8-27B_dev_20261118T162630.json|1"
  "microsoft/phi-4||15|8|22|otr09_microsoft__phi-4_dev_20261119T093535.json|0"
  "allenai/Olmo-3.1-32B-Instruct||18|8|26|otr09_allenai__Olmo-3.1-32B-Instruct_dev_20261119T095922.json|0"
  "ibm-granite/granite-4.2-30b|--no-thinking|37|28|52|otr09_ibm-granite__granite-4.2-30b_dev_20261119T140015.json|1"
)

run() {  # $1 = label, rest = sbatch arguments
  local label="$1"; shift
  printf '%s:' "$label"; printf ' %q' sbatch -A "$ACCT" --gpus=h100:1 "$@"; echo
  if [[ $SUBMIT == 1 ]]; then
    local out; out=$(sbatch -A "$ACCT" --gpus=h100:1 "$@")
    echo "  $out"
    echo "| $(date '+%Y-%m-%d %H:%M') | $COMMIT | $label | ${out##* } | submitted |" >> "$LOG"
  fi
}

even_upto() { local s=""; for ((l=0; l<=$1; l+=2)); do s+="${s:+,}$l"; done; echo "$s"; }

for row in "${MODELS[@]}"; do
  IFS='|' read -r M X ROLE BY HAND RF C7 <<< "$row"
  RFP="results/$RF"
  [[ -f "$RFP" ]] || { echo "missing $RFP (copy it from the results branch first)"; exit 1; }
  GRID09=$(even_upto $((HAND + 2)))
  XA=(); [[ -n "$X" ]] && XA=("$X")
  C=(--chat-template --split test0)
  run "C1 fail $M"  scripts/slurm/otrgen_pelle.sbatch "$M" "${XA[@]}" "${C[@]}" --n 512 --instruction "$INSTR" --question-template "$FAIL" --question-role agent
  run "C1 work $M"  scripts/slurm/otrgen_pelle.sbatch "$M" "${XA[@]}" "${C[@]}" --n 512 --instruction "$INSTR" --question-template "$WORK" --question-role agent
  run "C2 $M"       scripts/slurm/otr03_pelle.sbatch "$M" "${XA[@]}" "${C[@]}" --n-events 64 --layers "$ROLE"
  run "C3 $M"       scripts/slurm/otr03_pelle.sbatch "$M" "${XA[@]}" "${C[@]}" --n-events 64 --layers "$ROLE" --roles agent --agent-question "$FAIL" --include-incorrect
  run "C4/C5 $M"    scripts/slurm/otr09_pelle.sbatch "$M" "${XA[@]}" "${C[@]}" --instruction "$INSTR" --rank-from "$RFP" --skip-scan --n-random 5 --controls --layers "$GRID09"
  run "C6/C8 $M"    scripts/slurm/otr14_pelle.sbatch "$M" "${XA[@]}" "${C[@]}" --rank-from "$RFP" --layers "$BY"
  if [[ $C7 == 1 ]]; then
    run "C7 $M"     scripts/slurm/otr07_pelle.sbatch "$M" "${XA[@]}" "${C[@]}" --instruction "$INSTR" --by-prefix "Someone was {verb} by" --condition agent "$FAIL" --condition agent "$WORK"
  fi
done

# Replication on the broad English set (BX transitive), after the core batch: C2, C3, C5.
for row in "${MODELS[@]}"; do
  IFS='|' read -r M X ROLE BY HAND RF C7 <<< "$row"
  RFP="results/$RF"; GRID09=$(even_upto $((HAND + 2)))
  XA=(); [[ -n "$X" ]] && XA=("$X")
  C=(--chat-template --split test0 --stimuli bx-transitive)
  run "BX C2 $M"    scripts/slurm/otr03_pelle.sbatch "$M" "${XA[@]}" "${C[@]}" --n-events 48 --layers "$ROLE"
  run "BX C3 $M"    scripts/slurm/otr03_pelle.sbatch "$M" "${XA[@]}" "${C[@]}" --n-events 48 --layers "$ROLE" --agent-question "$FAIL" --include-incorrect
  run "BX C5 $M"    scripts/slurm/otr09_pelle.sbatch "$M" "${XA[@]}" "${C[@]}" --instruction "$INSTR" --rank-from "$RFP" --skip-scan --n-random 5 --layers "$GRID09"
done
[[ $SUBMIT == 1 ]] && echo "Logged to $LOG. Now hold the pending dev jobs so test0 runs first." || echo "(dry run: nothing submitted)"
