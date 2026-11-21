#!/bin/bash
# WIKI-ROLES, step 2 (compute node, CPU only): parse the downloaded Wikipedia shards and extract
# named-person transitive sentences. Run from the repository root after scripts/wiki_download.sh:
#   sbatch -A uppmax2026-1-225 scripts/slurm/wiki_extract_pelle.sbatch
#
#SBATCH -J wikiroles
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o results/slurm-wikiroles-%j.out

set -euo pipefail
source "${ORDER_ROLE_VENV:-/proj/${SLURM_JOB_ACCOUNT}/venvs/order-role}/bin/activate"
export HF_HUB_OFFLINE=1
mkdir -p results
python scripts/wiki_extract_roles.py --parquet "/proj/${SLURM_JOB_ACCOUNT}/wiki/20231101.en/train-*.parquet" \
    --out results/wiki_roles_en.jsonl --n-process 15 "$@"
python scripts/wiki_check_sheet.py results/wiki_roles_en.jsonl --n 200 --out results/wiki_check_sheet.csv
