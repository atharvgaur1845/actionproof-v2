#!/bin/bash
#SBATCH -p compute
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -J atk_smoke
#SBATCH --output=/scratch/pabitra/attacks/logs/%x-%j.out
cd /scratch/pabitra/attacks
export PYTHONDONTWRITEBYTECODE=1 HF_HOME=/scratch/pabitra/.cache/huggingface HF_HUB_OFFLINE=1 TMPDIR=/scratch/pabitra/attacks/.tmp
mkdir -p $TMPDIR
/scratch/pabitra/actionproof/.venv/bin/python -m pytest -q -p no:cacheprovider tests/attacks/test_model_loop_smoke.py -rs 2>&1 | tail -15
