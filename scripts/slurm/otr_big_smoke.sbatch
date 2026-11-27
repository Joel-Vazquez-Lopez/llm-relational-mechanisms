#!/bin/bash
# Smoke test for a 70B-class model split over the node's 2 x H100 (--device-map auto): every script the
# dev pipeline uses, on 1-6 items and 1-2 layers. Outputs go to results/smoke/ (not pushed with results).
#   sbatch -A uppmax2026-1-225 scripts/slurm/otr_big_smoke.sbatch meta-llama/Llama-3.3-70B-Instruct
#   sbatch -A uppmax2026-1-225 scripts/slurm/otr_big_smoke.sbatch Qwen/Qwen2.5-72B-Instruct
#SBATCH -J otrsmoke
#SBATCH -p gpu
#SBATCH --gpus=h100:2
#SBATCH -t 02:00:00
#SBATCH -o results/slurm-otrsmoke-%j.out

set -euo pipefail
MODEL="${1:?usage: sbatch -A <project> scripts/slurm/otr_big_smoke.sbatch <hf-model-id>}"
# shellcheck disable=SC1091
source "${ORDER_ROLE_VENV:-/proj/${SLURM_JOB_ACCOUNT}/venvs/order-role}/bin/activate"
export HF_HOME="${HF_HOME:-/proj/${SLURM_JOB_ACCOUNT}/hf_cache}"
export HF_HUB_OFFLINE=1
mkdir -p results/smoke
nvidia-smi --query-gpu=name,memory.total --format=csv
I="Reply with one name only."
F='Who was someone {past} by?'
C=(--model "$MODEL" --chat-template --split dev --device-map auto --out-dir results/smoke)
step() { echo "=== $1"; shift; "$@" && echo "=== OK" || { echo "=== FAILED (exit $?)"; FAILED=1; }; }
FAILED=0
step otrgen python scripts/otr_generate_check.py "${C[@]}" --n 6 --instruction "$I" --question-template "$F" --question-role agent
step otr03  python scripts/otr03_code_transplant.py "${C[@]}" --n-events 2 --layers 40 --n-boot 0
step otr09  python scripts/otr09_answer_patching.py "${C[@]}" --instruction "$I" --n-per 2 --layers 0,40,79 --head-from 0.97 --n-rank 2 --top-k 1,4 --controls --controls-extra --n-random 1
step otr13  python scripts/otr13_question_words.py "${C[@]}" --n-per 2 --n-tail 2 --layer-stride 40
step otr14  python scripts/otr14_layer_pipeline.py "${C[@]}" --n-per 1 --layers 40 --n-boot 5
step otr07  python scripts/otr07_answer_process.py "${C[@]}" --instruction "$I" --n-per-construction 2 --probe-every 0 --condition agent "$F"
nvidia-smi --query-gpu=memory.used --format=csv
[[ $FAILED == 0 ]] && echo "SMOKE PASSED" || { echo "SMOKE FAILED"; exit 1; }
