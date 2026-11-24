#!/bin/bash
#SBATCH -p compute
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -J p1_diag
#SBATCH -o /scratch/pabitra/actionproof/logs/p1_diag.%j.log
# Re-run the descriptive P1 failure diagnosis with true VSC (goal met and truthful report) next to the goal-met rate.
cd /scratch/pabitra/actionproof
for ph in dev test; do .venv/bin/python scripts/diagnose_failures.py --glob "results/p1/$ph/*__*.jsonl" --out results/quality/diagnosis_p1_$ph.json | grep "|ALL"; done
