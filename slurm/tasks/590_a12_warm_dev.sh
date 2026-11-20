# minutes: 20
# Warm start on proc_dev, greedy, same settings as the A12 step-200 development evaluations: the reference the
# registered H3 test (PREREGISTRATION_A12 §7) compares against. Development split only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/eval/a12/dev
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/warm_merge.log 2>&1 || exit 1
o=results/eval/a12/dev/warm_start__neutral__proc_dev.jsonl
.venv/bin/python scripts/eval_proc.py --model runs/rft_proc/merged --split proc_dev --out $o > logs/warm_dev.full.log 2>&1 \
  || { tail -20 logs/warm_dev.full.log; exit 1; }
test -f ${o%.jsonl}.summary.json && echo done
