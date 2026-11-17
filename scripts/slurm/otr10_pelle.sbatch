#!/bin/bash
# OTR-10 incremental order/function/role probes on UPPMAX Pelle (one H100).
#
# Usage (from the repository root):
#   sbatch -A <your-project> scripts/slurm/otr10_pelle.sbatch google/gemma-4-31B-it --chat-template --langs en,es,gl
#   sbatch -A <your-project> scripts/slurm/otr10_pelle.sbatch Qwen/Qwen3.8-27B --chat-template --no-thinking --langs en,es,gl
# States (~18 GiB for a 30B model with en,es,gl at --layer-stride 2) are kept on the GPU
# when they fit next to the model, otherwise in host memory; force with --store cpu|cuda.
#
#SBATCH -J otr10
#SBATCH -p gpu
#SBATCH --gpus=h100:1
#SBATCH -t 04:00:00
#SBATCH -o results/slurm-otr10-%j.out

set -euo pipefail
MODEL="${1:?usage: sbatch -A <project> scripts/slurm/otr10_pelle.sbatch <hf-model-id> [extra args]}"
shift

# shellcheck disable=SC1091
source "${ORDER_ROLE_VENV:-/proj/${SLURM_JOB_ACCOUNT}/venvs/order-role}/bin/activate"
export HF_HOME="${HF_HOME:-/proj/${SLURM_JOB_ACCOUNT}/hf_cache}"
export HF_HUB_OFFLINE=1   # compute nodes read the cache populated on the login node

mkdir -p results
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/otr10_incremental_roles.py --model "$MODEL" "$@"
