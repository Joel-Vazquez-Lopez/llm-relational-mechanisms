#!/bin/bash
# Submit the frozen Spanish/Galician test0 addendum (docs/order_to_role/FREEZE_XL.md). Run ONCE, on a Pelle login
# node, from the repository root, at the commit that adds FREEZE_XL.md, and only when no other jobs are queued:
#     bash scripts/slurm/otr_xl_test0_submit.sh            # prints the commands only (dry run)
#     bash scripts/slurm/otr_xl_test0_submit.sh --submit   # submits them and logs the job IDs
# 4 jobs per model and language (XL-C1 generation, XL-C2 and XL-C3 transplants, XL-C4/C5/C5b repair) = 40 jobs,
# each on one H100. Job IDs are appended to docs/order_to_role/test0_attempts.md.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ACCT="uppmax2026-1-225"
SUBMIT=0; [[ "${1:-}" == "--submit" ]] && SUBMIT=1
LOG=docs/order_to_role/test0_attempts.md
COMMIT=$(git rev-parse --short HEAD)

# The addendum must be committed before anything runs: refuse if FREEZE_XL.md or the scorer differ from HEAD.
for f in docs/order_to_role/FREEZE_XL.md scripts/otr_freeze_score.py; do
  [[ -f "$f" ]] || { echo "missing $f"; exit 1; }
  git diff --quiet HEAD -- "$f" || { echo "$f has uncommitted changes; commit the freeze first"; exit 1; }
done

declare -A INSTR=([es]="Responde con un solo nombre." [gl]="Responde cun só nome.")

# model | extra flag | role layer (English, frozen) | es handover | gl handover | dev OTR-09 file (frozen heads)
MODELS=(
  "google/gemma-4-31B-it||28|46|46|otr09_google__gemma-4-31B-it_dev_20261118T164326.json"
  "Qwen/Qwen3.8-27B|--no-thinking|29|50|50|otr09_Qwen__Qwen3.8-27B_dev_20261118T162630.json"
  "microsoft/phi-4||15|22|22|otr09_microsoft__phi-4_dev_20261119T093535.json"
  "allenai/Olmo-3.1-32B-Instruct||18|24|26|otr09_allenai__Olmo-3.1-32B-Instruct_dev_20261119T095922.json"
  "ibm-granite/granite-4.2-30b|--no-thinking|37|52|52|otr09_ibm-granite__granite-4.2-30b_dev_20261119T140015.json"
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

# Spanish first (primary), then Galician (secondary): same order in the queue.
for LANG in es gl; do
  for row in "${MODELS[@]}"; do
    IFS='|' read -r M X ROLE HES HGL RF <<< "$row"
    RFP="results/$RF"
    [[ -f "$RFP" ]] || { echo "missing $RFP (copy it from the results branch first)"; exit 1; }
    HAND=$HES; [[ $LANG == gl ]] && HAND=$HGL
    GRID09=$(even_upto $((HAND + 2)))
    XA=(); [[ -n "$X" ]] && XA=("$X")
    C=(--chat-template --split test0 --stimuli "$LANG")
    I="${INSTR[$LANG]}"
    run "XL-C1 $LANG $M"       scripts/slurm/otrgen_pelle.sbatch "$M" "${XA[@]}" "${C[@]}" --n 1640 --instruction "$I"
    run "XL-C2 $LANG $M"       scripts/slurm/otr03_pelle.sbatch "$M" "${XA[@]}" "${C[@]}" --n-events 64 --layers "$ROLE"
    run "XL-C3 $LANG $M"       scripts/slurm/otr03_pelle.sbatch "$M" "${XA[@]}" "${C[@]}" --n-events 64 --layers "$ROLE" --question-voices passive --roles agent --include-incorrect
    run "XL-C4/C5 $LANG $M"    scripts/slurm/otr09_pelle.sbatch "$M" "${XA[@]}" "${C[@]}" --instruction "$I" --rank-from "$RFP" --skip-scan --n-random 5 --controls --controls-extra --n-per 64 --top-k 20 --layers "$GRID09"
  done
done
[[ $SUBMIT == 1 ]] && echo "Logged to $LOG." || echo "(dry run: nothing submitted)"
