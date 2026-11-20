# minutes: 30
# Exploratory before/after reference (DECISIONS 2026-11-20 13:25): the warm start once on each confirmatory split,
# after every A12 model was frozen and evaluated. Not part of any registered test.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/eval/a12/conf_reference
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/ref_merge.log 2>&1 || exit 1
for sp in proc_conf proc_combo proc_stress; do
  .venv/bin/python scripts/eval_proc.py --model runs/rft_proc/merged --split $sp --confirmatory \
    --out results/eval/a12/conf_reference/warm_start__neutral__$sp.jsonl > logs/ref_$sp.log 2>&1 || { tail -20 logs/ref_$sp.log; exit 1; }
done
ls results/eval/a12/conf_reference/
