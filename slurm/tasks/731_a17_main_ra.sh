# minutes: 240
# A17 registered run (PREREGISTRATION_A17), part 2: R_A seeds 0-4: both OpsSim domains, main splits and A15 cross-domain forks.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_a17.py --check || exit 1
mkdir -p logs results/eval/a17
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/a17_merge.log 2>&1 || { tail -20 logs/a17_merge.log; exit 1; }

for tag in ra_s0 ra_s1 ra_s2 ra_s3 ra_s4; do
  ad=""; [ $tag = warm_start ] || ad="--adapter runs/a12_proc_$tag/final"
  for d in it_ops care_admin; do
    for m in main fork; do
      sp=$d; [ $m = fork ] && sp=${d}_cf
      s=results/eval/a17/${tag}__${sp}.summary.json
      [ -f $s ] || .venv/bin/python scripts/eval_opsim.py --domain $d --mode $m --model runs/rft_proc/merged $ad --tag $tag \
        > logs/a17_${tag}_$sp.full.log 2>&1 || { tail -20 logs/a17_${tag}_$sp.full.log; exit 1; }
      test -f $s || { echo "missing $s"; exit 1; }
    done
  done
  rm -rf runs/a12_proc_$tag/merged; echo "done $tag"
done
