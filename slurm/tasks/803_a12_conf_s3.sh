# minutes: 90
# A12 confirmatory evaluation, seed 3 (addendum 2 §2): each frozen step-200 adapter once on proc_conf, proc_combo,
# proc_stress, through scripts/conf_eval.py (HF-logged once-only access, per-task resume).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_models.py --check --tags a12_proc_rv_s3,a12_proc_ra_s3 || exit 1
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/conf_merge.log 2>&1 || exit 1
mkdir -p logs
for arm in proc_rv proc_ra; do
  for sp in proc_conf proc_combo proc_stress; do
    .venv/bin/python scripts/conf_eval.py --tag a12_${arm}_s3 --split $sp > logs/conf_${arm}_s3_$sp.full.log 2>&1 \
      || { tail -20 logs/conf_${arm}_s3_$sp.full.log; exit 1; }
    tail -1 logs/conf_${arm}_s3_$sp.full.log
  done
done
