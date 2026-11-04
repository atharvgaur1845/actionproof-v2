#!/bin/bash
# One GRPO arm with the shared protocol (spec §13.3): only ARM differs between arms of a round.
#   slurm/grpo_arm.sh ARM INIT_RUN BAND TAG     e.g. pcstr_d runs/rft_tiny_v2 data/curriculum/band_tiny_v2.jsonl v2
# INIT_RUN/final is merged once (INIT_RUN/merged) and every arm trains a fresh LoRA on it.
set -euo pipefail
ARM=$1; INIT=$2; BAND=$3; TAG=$4
O=runs/grpo_tiny_${TAG}_${ARM}_s0
L=logs/grpo_${TAG}_${ARM}
[ -f $O/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.grpo --arm $ARM --model CohereLabs/tiny-aya-global \
  --init-adapter $INIT/final --task-file $BAND --out $O \
  --steps 200 --num-generations 8 --prompts-per-step 4 --save-steps 25 --seed 0 --use-vllm --rollout batched \
  > $L.full.log 2>&1
tail -c 600 $L.full.log | tr -d '\r' | tail -2
mkdir -p results/eval/r3
for s in val test_iid test_ood_fault; do
  R=results/eval/r3/grpo_tiny_${TAG}_${ARM}_s0__neutral__$s
  [ -f $R.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval --model $INIT/merged --adapter $O/final --split $s \
    --prompt neutral --out $R.jsonl > ${L}_eval_$s.full.log 2>&1
  python3 -c "import json;d=json.load(open('$R.summary.json'));print('$TAG $ARM', '$s', {k:round(d[k],3) for k in ('vsc','fcr','ivr','dsr','report_rate','verify_calls')})"
done
