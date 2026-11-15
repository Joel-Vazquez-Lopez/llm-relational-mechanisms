#!/bin/bash
# Download Hugging Face models into the project cache on a CPU node (compute nodes have internet).
#   DL=$(sbatch --parsable -A uppmax2026-1-225 scripts/slurm/otr_download.sbatch <model> [<model> ...]); DL=${DL%%;*}
#   sbatch -A uppmax2026-1-225 --dependency=afterany:$DL ...    # chain jobs after the download
#SBATCH -J hf-download
#SBATCH -n 2
#SBATCH -t 04:00:00
#SBATCH -o results/slurm-download-%j.out
source /proj/uppmax2026-1-225/venvs/order-role/bin/activate
export HF_HOME=/proj/uppmax2026-1-225/hf_cache
for m in "$@"; do
  python -c "from huggingface_hub import snapshot_download as d; d('$m', allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.model','*.jinja'], ignore_patterns=['consolidated*','original/*'])" && echo "OK $m" || echo "FAILED $m"
done
