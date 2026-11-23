# P1 evaluation set (PREREGISTRATION_P1.md §5).
#   usage: source slurm/p1_evalset.sh <phase> "<tag>|<base model>|<adapter or ->|<recon 0/1> ..." "<split> ..."
PHASE=$1; MODELS=$2; SPLITS=$3
.venv/bin/python scripts/freeze_p1.py --check || exit 1
mkdir -p logs results/p1/$PHASE
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/p1_merge_rft.log 2>&1 || exit 1
for m in $MODELS; do
  IFS='|' read -r tag base ad recon <<< "$m"
  A=""; [ "$ad" != "-" ] && A="--adapter $ad"; R=""; [ "$recon" = 1 ] && R="--recon"
  for sp in $SPLITS; do
    [ -f results/p1/$PHASE/${tag}__$sp.summary.json ] && { echo "skip $tag $sp"; continue; }
    .venv/bin/python scripts/p1_eval.py --model $base $A --tag $tag --split $sp --phase $PHASE --samples 4 $R \
      > logs/p1_${PHASE}_${tag}_$sp.log 2>&1 || { tail -30 logs/p1_${PHASE}_${tag}_$sp.log; exit 1; }
    tail -1 logs/p1_${PHASE}_${tag}_$sp.log
  done
done
