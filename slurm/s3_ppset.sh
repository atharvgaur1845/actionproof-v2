# S3 addendum 2 (PREREGISTRATION_S3_ADDENDUM2 Part B): each model once on the targeted partial-payment splits.
#   usage (from a task script): source slurm/s3_ppset.sh "tag=adapter_or_empty ..."
.venv/bin/python scripts/freeze_s3a2.py --check || exit 1
mkdir -p logs results/s3/pp
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/s3pp_merge.log 2>&1 || { tail -20 logs/s3pp_merge.log; exit 1; }
for m in $1; do
  tag=${m%%=*}; ad=${m#*=}; A=""; [ -n "$ad" ] && A="--adapter $ad"
  for sp in s3_pp_full s3_pp_rec; do
    s=results/s3/pp/${tag}__${sp}.summary.json
    [ -f $s ] || .venv/bin/python scripts/s3_eval.py --model runs/rft_proc/merged $A --tag $tag --split $sp \
      --out-dir results/s3/pp > logs/s3pp_${tag}_$sp.log 2>&1 || { tail -20 logs/s3pp_${tag}_$sp.log; exit 1; }
  done
  [ -n "$ad" ] && rm -rf $(dirname $ad)/merged
  echo "done $tag"
done
