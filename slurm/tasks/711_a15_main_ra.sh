# minutes: 240
# A15 registered run (PREREGISTRATION_A15), part 2: R_A seeds 0-4.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29711
.venv/bin/python scripts/freeze_a15.py --check || exit 1
mkdir -p logs results/a15 results/a16 results/eval/a15
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final CohereLabs/tiny-aya-global > logs/a15_merge.log 2>&1 || { tail -20 logs/a15_merge.log; exit 1; }
for tag in ra_s0 ra_s1 ra_s2 ra_s3 ra_s4; do
  ad=""; [ $tag = warm_start ] || ad="--adapter runs/a12_proc_$tag/final"
  s=results/eval/a15/summary__${tag}__proc_mech.json
  [ -f $s ] || .venv/bin/python scripts/a15_eval.py model --model runs/rft_proc/merged $ad --tag $tag \
    > logs/a15_$tag.full.log 2>&1 || { tail -20 logs/a15_$tag.full.log; exit 1; }
  test -f $s || { echo "missing $s"; exit 1; }
  rm -rf runs/a12_proc_$tag/merged; echo "done $tag"
done
