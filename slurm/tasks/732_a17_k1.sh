# minutes: 120
# A17 secondary (PREREGISTRATION_A17 §3; K1 §3): the six frozen K1 models on both OpsSim main splits.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29732
.venv/bin/python scripts/freeze_a17.py --check && .venv/bin/python scripts/freeze_k1_models.py --check || exit 1
mkdir -p logs results/eval/a17
TAGS=$(python3 -c "import json;print(' '.join(sorted(json.load(open('results/k1/frozen_models.json')))))")
for tag in $TAGS; do
  for d in it_ops care_admin; do
    s=results/eval/a17/${tag}__${d}.summary.json
    [ -f $s ] || .venv/bin/python scripts/eval_opsim.py --domain $d --mode main --model runs/rft_proc/merged \
      --adapter runs/$tag/final --tag $tag > logs/a17_${tag}_$d.full.log 2>&1 || { tail -20 logs/a17_${tag}_$d.full.log; exit 1; }
  done
  rm -rf runs/$tag/merged; echo "done $tag"
done
