# AD-C1 training arm (PREREGISTRATION_AD_C1.md §2): seeds 2-4, training only (no development evaluation).
#   usage (from a task script): source slurm/ad_arm_c1.sh <estimator> <seed>
EST=$1; SEED=$2; OUT=runs/ad_${EST}_s${SEED}; TAG=ad_${EST}_s${SEED}
.venv/bin/python scripts/ad/freeze_c1.py --check || exit 1
case $EST in
  *_ema) .venv/bin/python scripts/ad/freeze_d2.py --check || exit 1; MOD=actionproof.ad.train_ema ;;
  *)     .venv/bin/python scripts/ad/freeze.py --check || exit 1; MOD=actionproof.ad.train ;;
esac
mkdir -p runs logs results/ad/rollouts results/ad/train_state
export AP_ROLLOUT_LOG=results/ad/rollouts/${TAG}.jsonl
[ -f $OUT/final/adapter_config.json ] || .venv/bin/python -m $MOD --estimator $EST \
  --pools data/bench/proc/cw_train.jsonl --seed $SEED --steps 200 --init-adapter runs/rft_proc/final --out $OUT \
  --use-vllm > logs/${TAG}.full.log 2>&1 || { tail -40 logs/${TAG}.full.log; exit 1; }
[ -f $AP_ROLLOUT_LOG ] && gzip -f $AP_ROLLOUT_LOG
cp $OUT/log_history.json results/ad/train_state/${TAG}.json
[ -f $OUT/ema_scale.jsonl ] && cp $OUT/ema_scale.jsonl results/ad/train_state/${TAG}.ema_scale.jsonl
rm -rf $OUT/checkpoint-*/optimizer.pt $OUT/merged
echo TRAINED $TAG
