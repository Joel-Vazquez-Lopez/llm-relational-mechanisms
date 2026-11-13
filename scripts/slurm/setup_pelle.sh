#!/bin/bash
# One-time setup for the order-to-role experiments on UPPMAX Pelle.
# Run on a LOGIN node (it needs internet), from the repository root:
#
#   bash scripts/slurm/setup_pelle.sh <your-project>            # e.g. uppmax2025-2-123
#   bash scripts/slurm/setup_pelle.sh <your-project> --no-models # skip model downloads
#
# Everything large goes into /proj/<project>/ (home quota is 64 GB):
#   /proj/<project>/venvs/order-role   Python environment (~6 GB)
#   /proj/<project>/hf_cache           model weights (Qwen2.5-1.5B ~3 GB, Qwen2.5-7B ~15 GB)
# Neither is inside /proj/<project>/backup, so neither is backed up (both can be re-created).

set -euo pipefail

PROJECT="${1:?usage: bash scripts/slurm/setup_pelle.sh <your-project> [--no-models]}"
DOWNLOAD_MODELS=1
[[ "${2:-}" == "--no-models" ]] && DOWNLOAD_MODELS=0

P="/proj/${PROJECT}"
VENV="${P}/venvs/order-role"
CACHE="${P}/hf_cache"

if [[ ! -d "$P" ]]; then
  echo "ERROR: $P does not exist. Check the project name with: projinfo" >&2
  exit 1
fi
if [[ ! -f scripts/ml101_order_role_behavior.py ]]; then
  echo "ERROR: run this from the repository root." >&2
  exit 1
fi

echo "== Storage before setup"
uquota || true

echo "== Python"
PY=python3
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "ERROR: $("$PY" --version) is too old; need 3.10+." >&2
  echo "Run 'module spider Python', load a 3.10+ module (module load Python/<version>), then rerun." >&2
  exit 1
fi
"$PY" --version

echo "== Virtual environment in $VENV"
mkdir -p "$(dirname "$VENV")"
[[ -d "$VENV" ]] || "$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch transformers accelerate matplotlib huggingface_hub

echo "== Model cache in $CACHE"
mkdir -p "$CACHE"
LINE="export HF_HOME=${CACHE}"
grep -qxF "$LINE" ~/.bashrc 2>/dev/null || echo "$LINE" >> ~/.bashrc
export HF_HOME="$CACHE"

echo "== Unit tests (no GPU or downloads needed)"
PYTHONPATH=src python -m unittest discover -s tests 2>&1 | tail -3

if [[ "$DOWNLOAD_MODELS" == 1 ]]; then
  echo "== Downloading Qwen2.5-1.5B and Qwen2.5-7B (one weight format only)"
  python - <<'EOF'
from huggingface_hub import snapshot_download

for repo in ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-7B"):
    path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model"])
    print(repo, "->", path)
EOF
fi

mkdir -p results
echo "== Storage after setup"
uquota || true
cat <<EOF

Setup done. Submit the first jobs from the repository root:
  sbatch -A ${PROJECT} scripts/slurm/ml101_pelle.sbatch Qwen/Qwen2.5-1.5B
  sbatch -A ${PROJECT} scripts/slurm/ml101_pelle.sbatch Qwen/Qwen2.5-7B
  sbatch -A ${PROJECT} scripts/slurm/ml102_pelle.sbatch Qwen/Qwen2.5-7B
Watch them with:  squeue -u \$USER
Logs and results: results/
EOF
