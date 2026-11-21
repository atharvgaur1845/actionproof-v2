# minutes: 300
# after: 983_id1d_warmstart
# ID1 (PREREGISTRATION_ID1): R_V GRPO from the no-verification warm start, seed 0 (A12 settings); dev diagnostics;
# proc_conf once.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29981
.venv/bin/python scripts/freeze_id1.py --check || exit 1
.venv/bin/python scripts/freeze_id1_d1.py --check || exit 1   # deviation D-ID1-1: proc_conf needs --confirmatory
OUT=runs/id1_rv_s0; TAG=id1_rv_s0
mkdir -p logs results/id1/rollouts results/id1/dev
export AP_ROLLOUT_LOG=results/id1/rollouts/${TAG}.jsonl
[ -f $OUT/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.grpo_proc --arm proc_rv --seed 0 \
  --init-adapter runs/rft_proc_v0/final --task-jsonl data/bench/proc/proc_train_b.jsonl --out $OUT --use-vllm \
  > logs/${TAG}.full.log 2>&1 || { tail -40 logs/${TAG}.full.log; exit 1; }
[ -f $AP_ROLLOUT_LOG ] && gzip -f $AP_ROLLOUT_LOG
rm -rf $OUT/checkpoint-*/optimizer.pt
for st in 100 200; do
  o=results/id1/dev/${TAG}_step${st}__neutral__proc_dev.jsonl
  [ -f ${o%.jsonl}.summary.json ] || .venv/bin/python scripts/eval_proc.py --model runs/rft_proc_v0/merged \
    --adapter $OUT/checkpoint-$st --split proc_dev --out $o > logs/${TAG}_dev.log 2>&1 || { tail -20 logs/${TAG}_dev.log; exit 1; }
  rm -rf $OUT/checkpoint-${st}_merged
done
C=results/id1/${TAG}__neutral__proc_conf.jsonl
[ -f ${C%.jsonl}.summary.json ] || .venv/bin/python scripts/eval_proc.py --model runs/rft_proc_v0/merged \
  --adapter $OUT/final --split proc_conf --confirmatory --out $C > logs/${TAG}_conf.log 2>&1 || { tail -20 logs/${TAG}_conf.log; exit 1; }
rm -rf $OUT/merged
tail -1 logs/${TAG}_conf.log
