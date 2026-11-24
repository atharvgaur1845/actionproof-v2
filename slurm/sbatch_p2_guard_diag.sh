#!/bin/bash
#SBATCH -p compute
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -J p2_gdiag
#SBATCH -o /scratch/pabitra/actionproof/logs/p2_gdiag.%j.log
# Descriptive diagnosis of the P2 guard (scripts/p2_guard_diagnosis.py) and P2 failure stages.
cd /scratch/pabitra/actionproof
source slurm/env.sh 2>/dev/null
.venv/bin/python scripts/p2_guard_diagnosis.py
.venv/bin/python scripts/diagnose_failures.py --glob "results/p2/test/*__p1_test_*.jsonl" --out results/quality/diagnosis_p2_test.json | grep "|ALL" | cut -c1-200
