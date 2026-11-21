# minutes: 40
# Stage 3 development reference: A12 R_V seed 1 (200 steps) on s3_rec_dev and proc_dev, for the 2-seed stage B comparison.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/s3/dev
for sp in s3_rec_dev proc_dev; do
  .venv/bin/python scripts/s3_eval.py --model runs/rft_proc/merged --adapter runs/a12_proc_rv_s1/final --tag a12_rv_s1 \
    --split $sp --n-tasks 150 > logs/s3_a12_rv_s1_$sp.log 2>&1 || { tail -20 logs/s3_a12_rv_s1_$sp.log; exit 1; }
  tail -1 logs/s3_a12_rv_s1_$sp.log
done
