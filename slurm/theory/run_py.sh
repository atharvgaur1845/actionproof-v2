#!/bin/bash
# Generic CPU runner for research/latent-state-theory jobs. Runs from /scratch/pabitra/lab_theory (own copy; never the
# hash-checked /scratch/pabitra/actionproof tree). Uses that tree's venv read-only.
#   sbatch -J <name> -c <cpus> -t <time> slurm/theory/run_py.sh <script.py> [args...]
#SBATCH -p compute
#SBATCH --mem=16G
#SBATCH -o /scratch/pabitra/lab_theory/logs/%x.%j.log
set -euo pipefail
cd /scratch/pabitra/lab_theory
export PYTHONPATH=/scratch/pabitra/lab_theory
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "host $(hostname) start $(date -Is) commit $(cat COMMIT 2>/dev/null)"
/scratch/pabitra/actionproof/.venv/bin/python "$@"
echo "end $(date -Is)"
