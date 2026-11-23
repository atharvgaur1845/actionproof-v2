# V2 development arm: training through actionproof/ad2/train.py, then the first-stage development evaluation
# (cw_dev greedy + 4 samples, proc_dev greedy), exactly as slurm/ad_arm.sh.
#   usage (from a task script): source slurm/ad2_arm.sh <registration> <estimator> <groups> <seed> <arm tag> [extra train args]
REG=$1; EST=$2; GROUPS=$3; SEED=$4; ARM=$5; shift 5; EXTRA="$*"
TAG=ad2_${ARM}_s${SEED}; OUT=runs/$TAG
.venv/bin/python scripts/ad2/freeze.py $REG --check || exit 1
mkdir -p runs logs results/ad2/rollouts results/ad2/dev results/ad2/train_state
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/ad_merge.log 2>&1 || { tail -20 logs/ad_merge.log; exit 1; }
export AP_ROLLOUT_LOG=results/ad2/rollouts/${TAG}.jsonl
[ -f $OUT/final/adapter_config.json ] || .venv/bin/python -m actionproof.ad2.train --groups $GROUPS --estimator $EST \
  --pools data/bench/proc/cw_train.jsonl --seed $SEED --steps 200 --init-adapter runs/rft_proc/final --out $OUT \
  --use-vllm $EXTRA > logs/${TAG}.full.log 2>&1 || { tail -40 logs/${TAG}.full.log; exit 1; }
[ -f $AP_ROLLOUT_LOG ] && gzip -f $AP_ROLLOUT_LOG
cp $OUT/log_history.json results/ad2/train_state/${TAG}.json
rm -rf $OUT/checkpoint-*/optimizer.pt
for sp in cw_dev proc_dev; do
  s=results/ad2/dev/${TAG}__${sp}.summary.json
  SAMP=0; [ $sp = cw_dev ] && SAMP=4
  [ -f $s ] || .venv/bin/python -m actionproof.ad.evaluate --model runs/rft_proc/merged --adapter $OUT/final --tag $TAG \
    --split $sp --n-tasks 150 --samples $SAMP --cost 0.05 --out-dir results/ad2/dev > logs/${TAG}_eval_$sp.log 2>&1 \
    || { tail -20 logs/${TAG}_eval_$sp.log; exit 1; }
  grep -v '^INFO\|^WARNING' logs/${TAG}_eval_$sp.log | tail -5
done
rm -rf $OUT/merged
echo AD2_ARM_OK $TAG
