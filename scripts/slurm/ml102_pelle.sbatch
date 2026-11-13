#!/bin/bash
# ML-102 position x layer patching map on UPPMAX Pelle (one L40s, 48 GB).
#
# Usage (from the repository root, after the one-time setup below):
#   sbatch -A <your-project> scripts/slurm/ml102_pelle.sbatch Qwen/Qwen2.5-1.5B
#   sbatch -A <your-project> scripts/slurm/ml102_pelle.sbatch meta-llama/Llama-3.2-3B
#   sbatch -A <your-project> scripts/slurm/ml102_pelle.sbatch Qwen/Qwen2.5-7B
# For models above ~20B parameters use an H100 instead:
#   sbatch -A <your-project> --gpus=h100:1 scripts/slurm/ml102_pelle.sbatch <model>
#
# One-time setup on a login node (login nodes have internet; download weights there).
# Keep everything large in the PROJECT folder, never in $HOME (64 GB quota on Pelle):
#   P=/proj/<your-project>
#   python3 -m venv $P/venvs/order-role && source $P/venvs/order-role/bin/activate
#   pip install --no-cache-dir torch transformers accelerate matplotlib
#   export HF_HOME=$P/hf_cache            # add to ~/.bashrc; outside $P/backup, so not backed up
#   huggingface-cli login                 # only for gated models such as Llama or Gemma
#   huggingface-cli download Qwen/Qwen2.5-7B --include "*.json" "*.safetensors" "tokenizer*" "*.txt"
#     (the --include filter skips duplicate weight formats, e.g. Llama's original/ folder)
#   uquota                                # check usage before and after each download
#
#SBATCH -J ml102
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -t 04:00:00
#SBATCH -o results/slurm-ml102-%j.out

set -euo pipefail
MODEL="${1:?usage: sbatch -A <project> scripts/slurm/ml102_pelle.sbatch <hf-model-id> [extra args]}"
shift

source "${ORDER_ROLE_VENV:-/proj/${SLURM_JOB_ACCOUNT}/venvs/order-role}/bin/activate"
export HF_HOME="${HF_HOME:-/proj/${SLURM_JOB_ACCOUNT}/hf_cache}"
export HF_HUB_OFFLINE=1   # compute nodes read the cache populated on the login node

mkdir -p results
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/ml102_position_layer_patching.py --model "$MODEL" --split dev --plot "$@"
