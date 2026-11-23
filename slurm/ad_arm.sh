# Algorithm-discovery development arm (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL.md) + dev evaluation.
#   usage (from a task script): source slurm/ad_arm.sh <estimator> <seed> <steps>
EST=$1; SEED=$2; STEPS=$3; OUT=runs/ad_${EST}_s${SEED}; TAG=ad_${EST}_s${SEED}
.venv/bin/python scripts/ad/freeze.py --check || exit 1
mkdir -p runs logs results/ad/rollouts results/ad/dev results/ad/train_state
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/ad_merge.log 2>&1 || { tail -20 logs/ad_merge.log; exit 1; }
export AP_ROLLOUT_LOG=results/ad/rollouts/${TAG}.jsonl
[ -f $OUT/final/adapter_config.json ] || .venv/bin/python -m actionproof.ad.train --estimator $EST \
  --pools data/bench/proc/cw_train.jsonl --seed $SEED --steps $STEPS --init-adapter runs/rft_proc/final --out $OUT \
  --use-vllm > logs/${TAG}.full.log 2>&1 || { tail -40 logs/${TAG}.full.log; exit 1; }
[ -f $AP_ROLLOUT_LOG ] && gzip -f $AP_ROLLOUT_LOG
cp $OUT/log_history.json results/ad/train_state/${TAG}.json
rm -rf $OUT/checkpoint-*/optimizer.pt
for sp in cw_dev proc_dev; do
  s=results/ad/dev/${TAG}__${sp}.summary.json
  SAMP=0; [ $sp = cw_dev ] && SAMP=4
  [ -f $s ] || .venv/bin/python -m actionproof.ad.evaluate --model runs/rft_proc/merged --adapter $OUT/final --tag $TAG \
    --split $sp --n-tasks 150 --samples $SAMP --out-dir results/ad/dev > logs/${TAG}_eval_$sp.log 2>&1 \
    || { tail -20 logs/${TAG}_eval_$sp.log; exit 1; }
  grep -v '^INFO\|^WARNING' logs/${TAG}_eval_$sp.log | tail -5
done
rm -rf $OUT/merged
