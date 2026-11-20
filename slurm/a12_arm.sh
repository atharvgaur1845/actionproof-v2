# A12 GRPO run (PREREGISTRATION_A12 §5) + development diagnostics at steps 50/100/150/200 (§6; dev/val only).
#   usage (from a task script): source slurm/a12_arm.sh <proc_rv|proc_ra> <seed>
mkdir -p runs
ARM=$1; SEED=$2; OUT=runs/a12_${ARM}_s${SEED}; TAG=a12_${ARM}_s${SEED}
.venv/bin/python scripts/freeze_a12.py --check || exit 1
.venv/bin/python scripts/freeze_a12b.py --check || exit 1   # addendum 1: proc_train_b, gate at group size 8
python3 -c "import json,sys; sys.exit(0 if json.load(open('results/a12b/gate.json'))['pass'] else 1)" || { echo "gate not passed"; exit 1; }
test -f results/a12b/h3_predictions.json || { echo "H3 predictions missing"; exit 1; }
mkdir -p logs results/a12/rollouts results/eval/a12/dev
sha256sum results/a12b/h3_predictions.json results/a12b/gate.json runs/rft_proc/final/adapter_model.safetensors | tee $OUT.inputs.sha256 2>/dev/null || true
export AP_ROLLOUT_LOG=results/a12/rollouts/${TAG}.jsonl
[ -f $OUT/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.grpo_proc --arm $ARM --seed $SEED \
  --init-adapter runs/rft_proc/final --task-jsonl data/bench/proc/proc_train_b.jsonl --out $OUT --use-vllm > logs/${TAG}.full.log 2>&1 || { tail -40 logs/${TAG}.full.log; exit 1; }
gzip -f $AP_ROLLOUT_LOG
rm -rf $OUT/checkpoint-*/optimizer.pt
for st in 50 100 150 200; do
  ad=$OUT/checkpoint-$st
  for sp in proc_dev proc_val; do
    o=results/eval/a12/dev/${TAG}_step${st}__neutral__${sp}.jsonl
    [ -f ${o%.jsonl}.summary.json ] || .venv/bin/python scripts/eval_proc.py --model runs/rft_proc/merged --adapter $ad \
      --split $sp --out $o > logs/${TAG}_eval.full.log 2>&1 || { tail -20 logs/${TAG}_eval.full.log; exit 1; }
  done
  rm -rf ${ad}_merged
done
python3 - <<PY
import json
for sp in ("proc_dev","proc_val"):
    print("$TAG", sp, [round(json.load(open(f"results/eval/a12/dev/${TAG}_step{s}__neutral__{sp}.summary.json"))["phantom_readback"]["rate"] or 0, 3) for s in (50,100,150,200)])
PY
