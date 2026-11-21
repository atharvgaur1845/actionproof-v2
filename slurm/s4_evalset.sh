# S4-EVAL (PREREGISTRATION_S4_EVAL): each model once on every shift split.
#   usage (from a task script): source slurm/s4_evalset.sh "tag=adapter_or_empty ..."
.venv/bin/python scripts/freeze_s4eval.py --check || exit 1
mkdir -p logs results/s4/eval
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/s4e_merge.log 2>&1 || exit 1
for m in $1; do
  tag=${m%%=*}; ad=${m#*=}; A=""; [ -n "$ad" ] && A="--adapter $ad"
  for sp in s4_lang_en s4_lang_hi s4_lang_hinglish s4_lang_para s4_api s4_wf4 s4_combo; do
    s=results/s4/eval/${tag}__${sp}.summary.json
    [ -f $s ] || .venv/bin/python scripts/s4_eval.py --model runs/rft_proc/merged $A --tag $tag --split $sp \
      > logs/s4e_${tag}_$sp.log 2>&1 || { tail -20 logs/s4e_${tag}_$sp.log; exit 1; }
  done
  [ -n "$ad" ] && rm -rf $(dirname $ad)/merged
  echo "done $tag"
done
