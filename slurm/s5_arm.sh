# Stage 5 development arm (C5 or G) + development evaluation (docs/research/algorithm/C5G_DEV.md). Dev splits only.
#   usage (from a task script): source slurm/s5_arm.sh <mode> <pools> <seed> <steps>
NAME=$1; POOLS=$2; SEED=$3; STEPS=$4; OUT=runs/s5_${NAME}_s${SEED}; TAG=s5_${NAME}_s${SEED}
.venv/bin/python scripts/freeze_c5g.py --check || exit 1
mkdir -p runs logs results/s5/rollouts results/s5/dev
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/s5_merge.log 2>&1 || { tail -20 logs/s5_merge.log; exit 1; }
export AP_ROLLOUT_LOG=results/s5/rollouts/${TAG}.jsonl
[ -f $OUT/final/adapter_config.json ] || .venv/bin/python -m actionproof.stage4.train_c5g --mode $NAME --pools $POOLS --seed $SEED \
  --steps $STEPS --init-adapter runs/rft_proc/final --out $OUT --use-vllm > logs/${TAG}.full.log 2>&1 \
  || { tail -40 logs/${TAG}.full.log; exit 1; }
gzip -f $AP_ROLLOUT_LOG
cp $OUT/c5_groups.jsonl results/s5/${TAG}_c5_groups.jsonl 2>/dev/null
mkdir -p results/s5/train_state; cp $OUT/log_history.json results/s5/train_state/${TAG}.json
rm -rf $OUT/checkpoint-*/optimizer.pt
for sp in s3_rec_dev proc_dev; do
  s=results/s5/dev/${TAG}__${sp}.summary.json
  [ -f $s ] || .venv/bin/python scripts/s3_eval.py --model runs/rft_proc/merged --adapter $OUT/final --tag $TAG \
    --split $sp --n-tasks 150 --out-dir results/s5/dev > logs/${TAG}_eval_$sp.log 2>&1 || { tail -20 logs/${TAG}_eval_$sp.log; exit 1; }
  tail -1 logs/${TAG}_eval_$sp.log
done
rm -rf $OUT/merged
