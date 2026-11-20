# minutes: 150
# K1 evaluation (PREREGISTRATION_K1 §3): each frozen K1 model once on proc_conf, proc_combo, proc_stress through
# scripts/conf_eval.py (HF-logged once-only access, per-task resume). Models must be frozen (scripts/freeze_k1_models.py).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
TAGS=$(python3 -c "import json;print(' '.join(sorted(json.load(open('results/k1/frozen_models.json')))))")
.venv/bin/python scripts/freeze_k1_models.py --check || exit 1
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/conf_merge.log 2>&1 || exit 1
mkdir -p logs
for tag in $TAGS; do
  for sp in proc_conf proc_combo proc_stress; do
    .venv/bin/python scripts/conf_eval.py --tag $tag --split $sp --frozen-dir results/k1 --out-dir results/eval/k1/conf \
      > logs/conf_${tag}_$sp.full.log 2>&1 || { tail -20 logs/conf_${tag}_$sp.full.log; exit 1; }
    tail -1 logs/conf_${tag}_$sp.full.log
  done
  rm -rf runs/$tag/final_merged runs/$tag/merged
done
