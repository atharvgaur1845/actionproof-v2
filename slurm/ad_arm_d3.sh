# AD-D3 arm (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL_D3.md) + dev evaluation.
#   usage (from a task script): source slurm/ad_arm_d3.sh <estimator> <seed> <steps> [smoke]
EST=$1; SEED=$2; STEPS=$3; SMOKE=${4:-}
OUT=runs/ad_${EST}_s${SEED}; TAG=ad_${EST}_s${SEED}
.venv/bin/python scripts/ad/freeze_d3.py --check || exit 1
mkdir -p runs logs results/ad/rollouts results/ad/dev results/ad/train_state
if [ -n "$SMOKE" ] && [ ! -f $OUT/final/adapter_config.json ]; then
  rm -rf runs/ad_d3_smoke
  AP_ROLLOUT_LOG=logs/ad_d3_smoke.rollouts.jsonl .venv/bin/python -m actionproof.ad.train_irc --estimator $EST \
    --pools data/bench/proc/cw_train.jsonl --seed 0 --steps 2 --init-adapter runs/rft_proc/final --out runs/ad_d3_smoke \
    --use-vllm > logs/ad_d3_smoke.log 2>&1 || { tail -40 logs/ad_d3_smoke.log; exit 1; }
  n=$(wc -l < runs/ad_d3_smoke/irc_scale.jsonl); [ "$n" -ge 2 ] || { echo "irc log has $n rows"; exit 1; }
  cat runs/ad_d3_smoke/irc_scale.jsonl; rm -rf runs/ad_d3_smoke; echo D3_SMOKE_OK
fi
export AP_ROLLOUT_LOG=results/ad/rollouts/${TAG}.jsonl
[ -f $OUT/final/adapter_config.json ] || .venv/bin/python -m actionproof.ad.train_irc --estimator $EST \
  --pools data/bench/proc/cw_train.jsonl --seed $SEED --steps $STEPS --init-adapter runs/rft_proc/final --out $OUT \
  --use-vllm > logs/${TAG}.full.log 2>&1 || { tail -40 logs/${TAG}.full.log; exit 1; }
[ -f $AP_ROLLOUT_LOG ] && gzip -f $AP_ROLLOUT_LOG
cp $OUT/log_history.json results/ad/train_state/${TAG}.json
cp $OUT/irc_scale.jsonl results/ad/train_state/${TAG}.irc_scale.jsonl
rm -rf $OUT/checkpoint-*/optimizer.pt
for sp in cw_dev proc_dev; do
  s=results/ad/dev/${TAG}__${sp}.summary.json
  SAMP=0; [ $sp = cw_dev ] && SAMP=4
  [ -f $s ] || .venv/bin/python -m actionproof.ad.evaluate --model runs/rft_proc/merged --adapter $OUT/final --tag $TAG \
    --split $sp --n-tasks 150 --samples $SAMP --cost 0.05 --out-dir results/ad/dev > logs/${TAG}_eval_$sp.log 2>&1 \
    || { tail -20 logs/${TAG}_eval_$sp.log; exit 1; }
  grep -v '^INFO\|^WARNING' logs/${TAG}_eval_$sp.log | tail -5
done
rm -rf $OUT/merged
