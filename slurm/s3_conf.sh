# S3 confirmatory evaluation (PREREGISTRATION_S3 §3): each model once on every confirmatory split.
#   usage (from a task script): source slurm/s3_conf.sh "tag=adapter_or_empty ..."
.venv/bin/python scripts/freeze_s3.py --check || exit 1
mkdir -p logs results/s3/conf
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/s3c_merge.log 2>&1 || { tail -20 logs/s3c_merge.log; exit 1; }
for m in $1; do
  tag=${m%%=*}; ad=${m#*=}; A=""; [ -n "$ad" ] && A="--adapter $ad"
  for sp in s3_conf_rec s3_conf_full s3_conf_stress; do
    s=results/s3/conf/${tag}__${sp}.summary.json
    [ -f $s ] || .venv/bin/python scripts/s3_eval.py --model runs/rft_proc/merged $A --tag $tag --split $sp \
      --out-dir results/s3/conf > logs/s3c_${tag}_$sp.log 2>&1 || { tail -20 logs/s3c_${tag}_$sp.log; exit 1; }
  done
  for d in it_ops care_admin; do
    s=results/s3/conf/${tag}__s3_${d}.summary.json
    [ -f $s ] || .venv/bin/python scripts/eval_opsim.py --domain $d --split s3_$d --model runs/rft_proc/merged $A \
      --tag $tag --out-dir results/s3/conf > logs/s3c_${tag}_$d.log 2>&1 || { tail -20 logs/s3c_${tag}_$d.log; exit 1; }
  done
  [ -n "$ad" ] && rm -rf $(dirname $ad)/merged
  echo "done $tag"
done
