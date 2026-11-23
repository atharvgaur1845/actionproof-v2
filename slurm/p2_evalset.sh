# P2 evaluation set (PREREGISTRATION_P2.md §5).
#   usage: source slurm/p2_evalset.sh <phase> "<tag>|<base model>|<adapter or ->|<flags: - L G LG> ..." "<split> ..."
PHASE=$1; MODELS=$2; SPLITS=$3
.venv/bin/python scripts/freeze_p2.py --check || exit 1
mkdir -p logs results/p2/$PHASE
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/p2_merge_rft.log 2>&1 || exit 1
for m in $MODELS; do
  IFS='|' read -r tag base ad flags <<< "$m"
  A=""; [ "$ad" != "-" ] && A="--adapter $ad"
  F=""; [[ "$flags" == *L* ]] && F="$F --ledger"; [[ "$flags" == *G* ]] && F="$F --guard"
  for sp in $SPLITS; do
    [ -f results/p2/$PHASE/${tag}__$sp.summary.json ] && { echo "skip $tag $sp"; continue; }
    if [ "$sp" = s4_api ] && [[ "$flags" == *G* ]]; then echo "skip $tag $sp (guard undefined on renamed API)"; continue; fi
    .venv/bin/python scripts/p2_eval.py --model $base $A --tag $tag --split $sp --phase $PHASE --samples 4 $F \
      > logs/p2_${PHASE}_${tag}_$sp.log 2>&1 || { tail -30 logs/p2_${PHASE}_${tag}_$sp.log; exit 1; }
    tail -1 logs/p2_${PHASE}_${tag}_$sp.log
  done
done
