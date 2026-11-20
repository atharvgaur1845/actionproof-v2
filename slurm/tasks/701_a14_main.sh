# minutes: 120
# Amendment 14 registered run (PREREGISTRATION_A14): warm start and every frozen A12 step-200 adapter on proc_cf
# (250 tasks x 5 branches), primary representation (forced read in the agent's format) and the runtime-attributed
# ablation; scripted references on CPU. Runs only after scripts/freeze_a14b.py has frozen code, split and adapters.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_a14.py --check && .venv/bin/python scripts/freeze_a14b.py --check || exit 1
mkdir -p logs results/eval/a14
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/a14_merge.log 2>&1 || { tail -20 logs/a14_merge.log; exit 1; }
for rep in agent runtime; do
  .venv/bin/python scripts/cef_eval.py scripted --rep $rep --out-dir results/eval/a14 > logs/a14_scripted_$rep.log 2>&1 \
    || { tail -20 logs/a14_scripted_$rep.log; exit 1; }
done
TAGS=$(python3 -c "import json;print(' '.join(sorted(json.load(open('results/a12/frozen_models.json')))))")
for tag in warm_start $TAGS; do
  ad=""; [ $tag = warm_start ] || ad="--adapter runs/$tag/final"
  for rep in agent runtime; do
    s=results/eval/a14/summary__${tag}__proc_cf__${rep}.json
    [ -f $s ] || .venv/bin/python scripts/cef_eval.py model --model runs/rft_proc/merged $ad --tag $tag --rep $rep \
      --out-dir results/eval/a14 > logs/a14_${tag}_${rep}.full.log 2>&1 || { tail -20 logs/a14_${tag}_${rep}.full.log; exit 1; }
    test -f $s || { echo "missing $s"; exit 1; }
    echo "done $tag $rep"
  done
  rm -rf runs/$tag/merged
done
