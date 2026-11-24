#!/bin/bash
# TH1 shards as a Slurm array on compute (own copy /scratch/pabitra/lab_theory).
#   sbatch --array=0-31 --export=ALL,SPLIT=heldout,NSH=32 slurm/theory/th1_array.sh
#SBATCH -p compute
#SBATCH -c 8
#SBATCH -t 03:00:00
#SBATCH -J th1
#SBATCH -o /scratch/pabitra/lab_theory/logs/th1_%x_%A_%a.log
set -euo pipefail
cd /scratch/pabitra/lab_theory
export PYTHONPATH=/scratch/pabitra/lab_theory OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "host $(hostname) start $(date -Is) commit $(cat COMMIT 2>/dev/null) split $SPLIT shard $SLURM_ARRAY_TASK_ID/$NSH"
/scratch/pabitra/actionproof/.venv/bin/python scripts/theory/th1_run.py --split "$SPLIT" --shard "$SLURM_ARRAY_TASK_ID" \
  --nshards "$NSH" --workers 8
echo "end $(date -Is)"
