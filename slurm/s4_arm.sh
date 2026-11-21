# Stage 4 FCA arm + development evaluation (docs/research/algorithm/FCA_GRPO.md). Dev splits only.
#   usage (from a task script): source slurm/s4_arm.sh <name> <pools> <seed> <steps> <n_on>
NAME=$1; POOLS=$2; SEED=$3; STEPS=$4; NON=$5; OUT=runs/s4_${NAME}_s${SEED}; TAG=s4_${NAME}_s${SEED}
mkdir -p runs logs results/s4/rollouts results/s4/dev
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/s4_merge.log 2>&1 || { tail -20 logs/s4_merge.log; exit 1; }
export AP_ROLLOUT_LOG=results/s4/rollouts/${TAG}.jsonl
[ -f $OUT/final/adapter_config.json ] || .venv/bin/python -m actionproof.stage4.train_fca --pools $POOLS --seed $SEED \
  --steps $STEPS --n-on $NON --init-adapter runs/rft_proc/final --out $OUT --use-vllm > logs/${TAG}.full.log 2>&1 \
  || { tail -40 logs/${TAG}.full.log; exit 1; }
gzip -f $AP_ROLLOUT_LOG
cp $OUT/fca_groups.jsonl results/s4/${TAG}_fca_groups.jsonl 2>/dev/null
rm -rf $OUT/checkpoint-*/optimizer.pt
for sp in s3_rec_dev proc_dev; do
  s=results/s4/dev/${TAG}__${sp}.summary.json
  [ -f $s ] || .venv/bin/python scripts/s3_eval.py --model runs/rft_proc/merged --adapter $OUT/final --tag $TAG \
    --split $sp --n-tasks 150 --out-dir results/s4/dev > logs/${TAG}_eval_$sp.log 2>&1 || { tail -20 logs/${TAG}_eval_$sp.log; exit 1; }
  tail -1 logs/${TAG}_eval_$sp.log
done
rm -rf $OUT/merged
