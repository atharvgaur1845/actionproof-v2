# minutes: 40
# Track 3 latency (exploratory): batch-1 seconds per episode on 40 proc_dev tasks, warm start and seed-0 models.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/s2
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/lat_merge.log 2>&1 || exit 1
.venv/bin/python scripts/latency_proc.py model --model runs/rft_proc/merged --tag warm_start > logs/lat_warm.log 2>&1 || { tail -20 logs/lat_warm.log; exit 1; }
for arm in rv ra; do
  .venv/bin/python scripts/latency_proc.py model --model runs/rft_proc/merged --adapter runs/a12_proc_${arm}_s0/final \
    --tag ${arm}_s0 > logs/lat_${arm}.log 2>&1 || { tail -20 logs/lat_${arm}.log; exit 1; }
done
ls results/s2/latency_*.json
