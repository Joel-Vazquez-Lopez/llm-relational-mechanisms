#!/bin/bash
# OTR-16 ("by" -> handover heads -> answer mediation) on dev for all seven models, with each model's frozen "by"
# layer and frozen top-20 heads (FREEZE.md, FREEZE_70B.md). Exploratory: nothing here is pre-registered.
#     bash scripts/slurm/otr16_dev_submit.sh            # prints the commands only (dry run)
#     bash scripts/slurm/otr16_dev_submit.sh --submit   # submits them
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ACCT="uppmax2026-1-225"
SUBMIT=0; [[ "${1:-}" == "--submit" ]] && SUBMIT=1
INSTR="Reply with one name only."

# model | extra flags | "by" layer | dev OTR-09 file (frozen heads) | GPUs
MODELS=(
  "google/gemma-4-31B-it||28|otr09_google__gemma-4-31B-it_dev_20261118T164326.json|1"
  "Qwen/Qwen3.8-27B|--no-thinking|30|otr09_Qwen__Qwen3.8-27B_dev_20261118T162630.json|1"
  "microsoft/phi-4||8|otr09_microsoft__phi-4_dev_20261119T093535.json|1"
  "allenai/Olmo-3.1-32B-Instruct||8|otr09_allenai__Olmo-3.1-32B-Instruct_dev_20261119T095922.json|1"
  "ibm-granite/granite-4.2-30b|--no-thinking|28|otr09_ibm-granite__granite-4.2-30b_dev_20261119T140015.json|1"
  "meta-llama/Llama-3.3-70B-Instruct|--device-map auto|14|otr09_meta-llama__Llama-3.3-70B-Instruct_dev_20261127T121453.json|2"
  "Qwen/Qwen2.5-72B-Instruct|--device-map auto|30|otr09_Qwen__Qwen2.5-72B-Instruct_dev_20261127T121538.json|2"
)

for row in "${MODELS[@]}"; do
  IFS='|' read -r M X BY RF G <<< "$row"
  [[ -f "results/$RF" ]] || { echo "missing results/$RF"; exit 1; }
  read -r -a XA <<< "$X"
  CMD=(sbatch -A "$ACCT" --gpus="h100:$G" -t 04:00:00 scripts/slurm/otr16_pelle.sbatch "$M" "${XA[@]}"
       --chat-template --split dev --instruction "$INSTR" --by-layer "$BY" --rank-from "results/$RF" --n-random 5)
  printf 'OTR-16 %s:' "$M"; printf ' %q' "${CMD[@]}"; echo
  [[ $SUBMIT == 1 ]] && "${CMD[@]}"
done
[[ $SUBMIT == 1 ]] || echo "(dry run: nothing submitted)"
