# R1 capability retention (PREREGISTRATION_R1.md): generate IFEval + BFCL outputs for each model once.
#   usage (from a task script): source slurm/r1_evalset.sh "tag=run_dir ..."
#   special tags: base (the HF base model), warm_start (runs/rft_proc), s1_warm_start (runs/rft_tiny_v2).
#   Stage 1 tags (s1_*) merge onto runs/rft_tiny_v2/merged; every other adapter onto runs/rft_proc/merged.
.venv/bin/python scripts/freeze_r1.py --check || exit 1
mkdir -p logs results/r1/gen
for w in rft_proc rft_tiny_v2; do
  [ -f runs/$w/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/$w/final \
    CohereLabs/tiny-aya-global > logs/r1_merge_$w.log 2>&1 || exit 1
done
for m in $1; do
  tag=${m%%=*}; run=${m#*=}
  [ -f results/r1/gen/$tag.jsonl ] && { echo "skip $tag (done)"; continue; }
  made=""
  case $tag in
    base) model=CohereLabs/tiny-aya-global ;;
    warm_start) model=runs/rft_proc/merged ;;
    s1_warm_start) model=runs/rft_tiny_v2/merged ;;
    *)
      base=runs/rft_proc/merged; case $tag in s1_*) base=runs/rft_tiny_v2/merged ;; esac
      model=$run/merged
      if [ ! -f $model/config.json ]; then
        .venv/bin/python -m actionproof.train.merge $run/final $base > logs/r1_merge_$tag.log 2>&1 \
          || { tail -20 logs/r1_merge_$tag.log; exit 1; }
        made=1
      fi ;;
  esac
  .venv/bin/python -m actionproof.retention.gen --model $model --tag $tag ${R1_LIMIT:+--limit $R1_LIMIT} \
    ${R1_OUT:+--out-dir $R1_OUT} > logs/r1_gen_$tag.log 2>&1 || { tail -30 logs/r1_gen_$tag.log; exit 1; }
  [ -n "$made" ] && rm -rf $model   # remove only merges made here (cache; adapters untouched)
  echo "$(date +%T) done $tag"
done
