# K1 R_A GRPO run on a recomposed pool (PREREGISTRATION_K1). Same trainer, warm start and settings as A12 R_A;
# only --task-jsonl differs. Development diagnostics at steps 100/200 on proc_dev only.
#   usage (from a task script): source slurm/k1_arm.sh <k1t|k1p> <seed>
mkdir -p runs logs results/k1/rollouts results/eval/k1/dev
POOL=$1; SEED=$2; OUT=runs/k1_${POOL}_ra_s${SEED}; TAG=k1_${POOL}_ra_s${SEED}
.venv/bin/python scripts/freeze_a12.py --check || exit 1
.venv/bin/python scripts/freeze_k1.py --check || exit 1
sha256sum data/bench/proc/proc_train_${POOL}.jsonl runs/rft_proc/final/adapter_model.safetensors | tee $OUT.inputs.sha256 2>/dev/null || true
export AP_ROLLOUT_LOG=results/k1/rollouts/${TAG}.jsonl
[ -f $OUT/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.grpo_proc --arm proc_ra --seed $SEED \
  --init-adapter runs/rft_proc/final --task-jsonl data/bench/proc/proc_train_${POOL}.jsonl --out $OUT --use-vllm \
  > logs/${TAG}.full.log 2>&1 || { tail -40 logs/${TAG}.full.log; exit 1; }
gzip -f $AP_ROLLOUT_LOG
rm -rf $OUT/checkpoint-*/optimizer.pt
for st in 100 200; do
  o=results/eval/k1/dev/${TAG}_step${st}__neutral__proc_dev.jsonl
  [ -f ${o%.jsonl}.summary.json ] || .venv/bin/python scripts/eval_proc.py --model runs/rft_proc/merged \
    --adapter $OUT/checkpoint-$st --split proc_dev --out $o > logs/${TAG}_eval.full.log 2>&1 || { tail -20 logs/${TAG}_eval.full.log; exit 1; }
  rm -rf $OUT/checkpoint-${st}_merged
done
