#!/bin/bash
#SBATCH -p compute
#SBATCH -c 4
#SBATCH -t 03:00:00
#SBATCH -J claim_verify
#SBATCH -o /scratch/pabitra/actionproof/logs/claim_verify.%j.log
# Independent recomputation of headline claims (scripts/audit/claim_verify.py).
cd /scratch/pabitra/actionproof
.venv/bin/python scripts/audit/claim_verify.py --out results/audit/claim_verify.json
