# minutes: 60
# Stage 3 development baselines: warm start, A12 R_V seed 0 and R_A seed 0 on s3_rec_dev and proc_dev (150 tasks).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/s3/dev
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/s3_merge.log 2>&1 || exit 1
for m in "warm_start:" "a12_rv_s0:--adapter runs/a12_proc_rv_s0/final" "a12_ra_s0:--adapter runs/a12_proc_ra_s0/final"; do
  tag=${m%%:*}; ad=${m#*:}
  for sp in s3_rec_dev proc_dev; do
    .venv/bin/python scripts/s3_eval.py --model runs/rft_proc/merged $ad --tag $tag --split $sp --n-tasks 150 \
      > logs/s3_${tag}_$sp.log 2>&1 || { tail -20 logs/s3_${tag}_$sp.log; exit 1; }
    tail -1 logs/s3_${tag}_$sp.log
  done
done
