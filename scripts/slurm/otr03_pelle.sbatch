#!/bin/bash
# OTR-03 code transplant profile on UPPMAX Pelle (one L40s, 48 GB).
#
# Usage (from the repository root, after the one-time setup below):
#   sbatch -A <your-project> scripts/slurm/otr03_pelle.sbatch Qwen/Qwen2.5-1.5B
#   sbatch -A <your-project> scripts/slurm/otr03_pelle.sbatch meta-llama/Llama-3.2-3B
#   sbatch -A <your-project> scripts/slurm/otr03_pelle.sbatch Qwen/Qwen2.5-7B
# For models above ~20B parameters use an H100 instead:
#   sbatch -A <your-project> --gpus=h100:1 scripts/slurm/otr03_pelle.sbatch <model>
#
# One-time setup first (login node, from the repository root):
#   bash scripts/slurm/setup_pelle.sh <your-project>
#
#SBATCH -J otr03
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -t 04:00:00
#SBATCH -o results/slurm-otr03-%j.out

set -euo pipefail
MODEL="${1:?usage: sbatch -A <project> scripts/slurm/otr03_pelle.sbatch <hf-model-id> [extra args]}"
shift

# shellcheck disable=SC1091
source "${ORDER_ROLE_VENV:-/proj/${SLURM_JOB_ACCOUNT}/venvs/order-role}/bin/activate"
export HF_HOME="${HF_HOME:-/proj/${SLURM_JOB_ACCOUNT}/hf_cache}"
export HF_HUB_OFFLINE=1   # compute nodes read the cache populated on the login node

mkdir -p results
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/otr03_code_transplant.py --model "$MODEL" --split dev --plot "$@"
