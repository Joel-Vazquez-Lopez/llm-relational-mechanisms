#!/bin/bash
# WIKI-ROLES, step 1 (LOGIN node, has internet): install the parser and download Wikipedia shards
# into the project folder (not home). Run from the repository root:
#   bash scripts/wiki_download.sh uppmax2026-1-225 8
set -euo pipefail
PROJ="${1:?usage: bash scripts/wiki_download.sh <project> [n_shards]}"
N="${2:-8}"
source "${ORDER_ROLE_VENV:-/proj/${PROJ}/venvs/order-role}/bin/activate"
export HF_HOME="${HF_HOME:-/proj/${PROJ}/hf_cache}"
pip install -q "spacy>=3.7" lemminflect pyarrow
python -m spacy download en_core_web_lg
python - "$PROJ" "$N" <<'PY'
import sys
from huggingface_hub import snapshot_download
proj, n = sys.argv[1], int(sys.argv[2])
pats = [f"20231101.en/train-{i:05d}-of-*.parquet" for i in range(n)]
path = snapshot_download("wikimedia/wikipedia", repo_type="dataset", allow_patterns=pats,
                         local_dir=f"/proj/{proj}/wiki")
print("downloaded to", path)
PY
ls -la "/proj/${PROJ}/wiki/20231101.en/" | head
