#!/bin/bash
# OTR-12 head knockout + fetch-agent vector on UPPMAX Pelle (one H100).
#
# Usage (from the repository root):
#   sbatch -A <your-project> --mem=160G scripts/slurm/otr12_pelle.sbatch google/gemma-4-31B-it --chat-template --otr09 results/otr09_google__gemma-4-31B-it_dev_<stamp>.json
#
#SBATCH -J otr12
#SBATCH -p gpu
#SBATCH --gpus=h100:1
#SBATCH -t 04:00:00
#SBATCH -o results/slurm-otr12-%j.out

set -euo pipefail
MODEL="${1:?usage: sbatch -A <project> scripts/slurm/otr12_pelle.sbatch <hf-model-id> [extra args]}"
shift

# shellcheck disable=SC1091
source "${ORDER_ROLE_VENV:-/proj/${SLURM_JOB_ACCOUNT}/venvs/order-role}/bin/activate"
export HF_HOME="${HF_HOME:-/proj/${SLURM_JOB_ACCOUNT}/hf_cache}"
export HF_HUB_OFFLINE=1   # compute nodes read the cache populated on the login node

mkdir -p results
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/otr12_knockout_and_role_vector.py --model "$MODEL" "$@"
