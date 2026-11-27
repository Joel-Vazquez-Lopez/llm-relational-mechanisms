#!/bin/bash
# OTR-09 answer-side patching (repair the failing question) on UPPMAX Pelle (one H100).
#
# Usage (from the repository root):
#   sbatch -A <your-project> scripts/slurm/otr09_pelle.sbatch google/gemma-4-31B-it --chat-template --instruction "Reply with one name only." --heads 47.17,53.26,53.18,57.7
#   sbatch -A <your-project> scripts/slurm/otr09_pelle.sbatch Qwen/Qwen3.8-27B --chat-template --no-thinking --instruction "Reply with one name only."
#
#SBATCH -J otr09
#SBATCH -p gpu
#SBATCH --gpus=h100:1
#SBATCH -t 04:00:00
#SBATCH -o results/slurm-otr09-%j.out

set -euo pipefail
MODEL="${1:?usage: sbatch -A <project> scripts/slurm/otr09_pelle.sbatch <hf-model-id> [extra args]}"
shift

# shellcheck disable=SC1091
source "${ORDER_ROLE_VENV:-/proj/${SLURM_JOB_ACCOUNT}/venvs/order-role}/bin/activate"
export HF_HOME="${HF_HOME:-/proj/${SLURM_JOB_ACCOUNT}/hf_cache}"
export HF_HUB_OFFLINE=1   # compute nodes read the cache populated on the login node

mkdir -p results
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/otr09_answer_patching.py --model "$MODEL" "$@"
