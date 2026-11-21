# Stage 3 GRPO arm + development evaluation (docs/stage3/RESEARCH_REPORT.md). Dev splits only; never a test split.
#   usage (from a task script): source slurm/s3_arm.sh <name> <reward> <pools> <seed> <steps>
NAME=$1; REWARD=$2; POOLS=$3; SEED=$4; STEPS=$5; OUT=runs/s3_${NAME}_s${SEED}; TAG=s3_${NAME}_s${SEED}
mkdir -p runs logs results/s3/rollouts results/s3/dev
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/s3_merge.log 2>&1 || { tail -20 logs/s3_merge.log; exit 1; }
export AP_ROLLOUT_LOG=results/s3/rollouts/${TAG}.jsonl
[ -f $OUT/final/adapter_config.json ] || .venv/bin/python -m actionproof.stage3.train --reward $REWARD --pools $POOLS \
  --seed $SEED --steps $STEPS --init-adapter runs/rft_proc/final --out $OUT --use-vllm > logs/${TAG}.full.log 2>&1 \
  || { tail -40 logs/${TAG}.full.log; exit 1; }
gzip -f $AP_ROLLOUT_LOG
rm -rf $OUT/checkpoint-*/optimizer.pt
for sp in s3_rec_dev proc_dev; do
  s=results/s3/dev/${TAG}__${sp}.summary.json
  [ -f $s ] || .venv/bin/python scripts/s3_eval.py --model runs/rft_proc/merged --adapter $OUT/final --tag $TAG \
    --split $sp --n-tasks 150 > logs/${TAG}_eval_$sp.log 2>&1 || { tail -20 logs/${TAG}_eval_$sp.log; exit 1; }
  tail -1 logs/${TAG}_eval_$sp.log
done
rm -rf $OUT/merged
