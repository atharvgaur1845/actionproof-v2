# E1/E2 (PREREGISTRATION_E1E2): each model once on e1_pp_full, e1_pp_rec (8 samples + greedy) and the e2_pp probe.
#   usage (from a task script): source slurm/e1e2_set.sh "tag=adapter_or_empty ..."
.venv/bin/python scripts/freeze_e1e2.py --check || exit 1
mkdir -p logs results/e1 results/e2
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/e1e2_merge.log 2>&1 || { tail -20 logs/e1e2_merge.log; exit 1; }
for m in $1; do
  tag=${m%%=*}; ad=${m#*=}; M=runs/rft_proc/merged
  if [ -n "$ad" ]; then
    M=$(.venv/bin/python -c "from actionproof.train.merge import merge; print(merge('$ad', 'runs/rft_proc/merged'))" \
        2> logs/e1e2_merge_$tag.log | tail -1) || { tail -20 logs/e1e2_merge_$tag.log; exit 1; }
  fi
  for sp in e1_pp_full e1_pp_rec; do
    s=results/e1/${tag}__${sp}.summary.json
    [ -f $s ] || .venv/bin/python scripts/e1_eval.py --model $M --adapter "$ad" --tag $tag --split $sp --samples 8 \
      > logs/e1_${tag}_$sp.log 2>&1 || { tail -20 logs/e1_${tag}_$sp.log; exit 1; }
  done
  [ -f results/e2/$tag.summary.json ] || .venv/bin/python scripts/e2_probe.py --model $M --tag $tag \
    > logs/e2_$tag.log 2>&1 || { tail -20 logs/e2_$tag.log; exit 1; }
  [ -n "$ad" ] && rm -rf $(dirname $ad)/merged
  echo "done $tag"
done
