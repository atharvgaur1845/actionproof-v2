#!/bin/bash
# LAB1 CPU families (PREREGISTRATION_LAB1.md) on the Slurm compute partition, from /scratch/pabitra/lab_final only.
#SBATCH -J lab1_cpu
#SBATCH -p compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH -o /scratch/pabitra/lab_final/logs/%x_%j.out
set -uo pipefail
cd /scratch/pabitra/lab_final
export PYTHONPATH=/scratch/pabitra/lab_final OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/scratch/pabitra/actionproof/.venv/bin/python
date; hostname; echo "commit $(cat LAB_COMMIT 2>/dev/null)"
$PY scripts/lab/freeze_lab1.py --check || exit 1
for f in L1 L2N L3G; do $PY scripts/lab/lab1_sweep.py $f --workers 30 || exit 2; done
$PY scripts/lab/grad_snr.py --n 300 --workers 30 || exit 3
$PY scripts/lab/lab1_refs.py || exit 4
$PY scripts/lab/lab1_analyze.py || exit 5
date; echo LAB1_CPU_OK
