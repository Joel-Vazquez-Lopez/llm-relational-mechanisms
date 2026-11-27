#!/bin/bash
# OTR-14 layer pipeline (question read, names read, handover; per item with CIs) on UPPMAX Pelle (one H100).
#
# Usage (from the repository root):
#   sbatch -A <your-project> scripts/slurm/otr14_pelle.sbatch google/gemma-4-31B-it --chat-template --instruction "Reply with one name only."
#   sbatch -A <your-project> scripts/slurm/otr14_pelle.sbatch Qwen/Qwen3.8-27B --chat-template --no-thinking --instruction "Reply with one name only."
#
#SBATCH -J otr14
#SBATCH -p gpu
#SBATCH --gpus=h100:1
#SBATCH -t 04:00:00
#SBATCH -o results/slurm-otr14-%j.out

set -euo pipefail
MODEL="${1:?usage: sbatch -A <project> scripts/slurm/otr14_pelle.sbatch <hf-model-id> [extra args]}"
shift

# shellcheck disable=SC1091
source "${ORDER_ROLE_VENV:-/proj/${SLURM_JOB_ACCOUNT}/venvs/order-role}/bin/activate"
export HF_HOME="${HF_HOME:-/proj/${SLURM_JOB_ACCOUNT}/hf_cache}"
export HF_HUB_OFFLINE=1   # compute nodes read the cache populated on the login node

mkdir -p results
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/otr14_layer_pipeline.py --model "$MODEL" "$@"
