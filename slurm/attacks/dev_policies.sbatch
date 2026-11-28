#!/bin/bash
# Attack suite ATK1: unit tests, then the scripted-policy development evaluation (CPU only, no model, no GPU).
# Runs in a tar copy of the branch at /scratch/pabitra/attacks (never in the frozen /scratch/pabitra/actionproof).
#SBATCH -p compute
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -J atk_dev
#SBATCH --output=/scratch/pabitra/attacks/logs/%x-%j.out
set -uo pipefail
cd /scratch/pabitra/attacks
PY=/scratch/pabitra/actionproof/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
date
$PY -m pytest -q -p no:cacheprovider tests/attacks || exit 1
$PY scripts/attacks_policies_eval.py || exit 1
sha256sum results/attacks/dev_policies.*
date
