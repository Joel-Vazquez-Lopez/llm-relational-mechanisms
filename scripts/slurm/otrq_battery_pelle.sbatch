#!/bin/bash
#SBATCH -J otrq
#SBATCH -p gpu
#SBATCH --gpus=h100:1
#SBATCH -t 03:00:00
#SBATCH -o results/slurm-otrq-%j.out
# Question battery (dev): is the failure about the construction or about the placeholder "someone"?
# Naturalness check for the failing question. One job per model; 9 questions x n prompts.
#   sbatch -A <project> scripts/slurm/otrq_battery_pelle.sbatch <hf-model-id> [--no-thinking]
set -euo pipefail
MODEL="${1:?usage: sbatch -A <project> scripts/slurm/otrq_battery_pelle.sbatch <hf-model-id> [extra args]}"
shift
source "${ORDER_ROLE_VENV:-/proj/${SLURM_JOB_ACCOUNT}/venvs/order-role}/bin/activate"
export HF_HOME="${HF_HOME:-/proj/${SLURM_JOB_ACCOUNT}/hf_cache}"
export HF_HUB_OFFLINE=1
mkdir -p results
nvidia-smi --query-gpu=name,memory.total --format=csv
QS_AGENT=(
  "Who was {patient} {past} by?"      # natural: patient named, stranded by
  "By whom was {patient} {past}?"     # natural, formal: pied-piped
  "Who was someone {past} by?"        # the failing question (reference)
  "By whom was someone {past}?"       # placeholder, pied-piped: is final "by" needed?
  "Who {past} {patient}?"             # active wh-question, patient named
  "Who {past} someone?"               # active wh-question, placeholder
  "Who was the {ing} done by?"        # the working question (reference)
)
QS_PATIENT=(
  "Who did {agent} {lemma}?"
  "Who was {past} by {agent}?"
)
for q in "${QS_AGENT[@]}"; do r=agent; t="$q"; python scripts/otr_generate_check.py --model "$MODEL" --chat-template --split dev --n 256 --instruction "Reply with one name only." --question-role "$r" --question-template "$t" "$@" || echo "FAILED: $t"; done
for q in "${QS_PATIENT[@]}"; do r=patient; t="$q"; python scripts/otr_generate_check.py --model "$MODEL" --chat-template --split dev --n 256 --instruction "Reply with one name only." --question-role "$r" --question-template "$t" "$@" || echo "FAILED: $t"; done
