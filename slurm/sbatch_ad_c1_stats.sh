#!/bin/bash
#SBATCH -p compute
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -J ad_c1_stats
#SBATCH -o /scratch/pabitra/actionproof/logs/ad_c1_stats.%j.log
# AD-C1 registered analysis (PREREGISTRATION_AD_C1.md), run once after evals 1240/1241.
cd /scratch/pabitra/actionproof
.venv/bin/python scripts/ad/freeze_c1.py --check && .venv/bin/python scripts/ad/c1_stats.py --out results/ad/c1_summary.json
