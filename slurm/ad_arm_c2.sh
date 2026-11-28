# AD-C2 training arm (PREREGISTRATION_AD_C2.md): the AD-C1 recipe unchanged except --cost.
#   usage (from a task script): source slurm/ad_arm_c2.sh <estimator> <cost-tag> <cost> <seed>
EST=$1; CT=$2; COST=$3; SEED=$4; TAG=ad_c2_${EST}_${CT}_s${SEED}; OUT=runs/$TAG
.venv/bin/python scripts/ad/freeze_c2.py --check || exit 1
.venv/bin/python scripts/ad/freeze.py --check || exit 1
mkdir -p runs logs results/ad/rollouts results/ad/train_state
export AP_ROLLOUT_LOG=results/ad/rollouts/${TAG}.jsonl
[ -f $OUT/final/adapter_config.json ] || .venv/bin/python -m actionproof.ad.train --estimator $EST --cost $COST \
  --pools data/bench/proc/cw_train.jsonl --seed $SEED --steps 200 --init-adapter runs/rft_proc/final --out $OUT \
  --use-vllm > logs/${TAG}.full.log 2>&1 || { tail -40 logs/${TAG}.full.log; exit 1; }
[ -f $AP_ROLLOUT_LOG ] && gzip -f $AP_ROLLOUT_LOG
cp $OUT/log_history.json results/ad/train_state/${TAG}.json
rm -rf $OUT/checkpoint-*/optimizer.pt $OUT/merged
echo TRAINED $TAG
