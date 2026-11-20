# minutes: 45
# A17 pilot (development only, disclosed, not analysed): R_V seed 0 and the warm start on 12 pilot tasks per OpsSim
# domain (seeds 131/132, outside the registered splits): parsing, tool rendering, throughput.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/a17/pilot
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/a17p_merge.log 2>&1 || { tail -20 logs/a17p_merge.log; exit 1; }
.venv/bin/python -m pytest -q tests/integrity/test_opsim_it_ops.py tests/integrity/test_opsim_care_admin.py 2>&1 | tail -2
for d in it_ops care_admin; do
  for m in "warm_start:" "rv_s0:--adapter runs/a12_proc_rv_s0/final"; do
    tag=${m%%:*}; ad=${m#*:}
    .venv/bin/python scripts/eval_opsim.py --domain $d --split ${d}_pilot --model runs/rft_proc/merged $ad --tag $tag \
      --out-dir results/a17/pilot > logs/a17p_${d}_$tag.log 2>&1 || { tail -30 logs/a17p_${d}_$tag.log; exit 1; }
    tail -2 logs/a17p_${d}_$tag.log
  done
done
