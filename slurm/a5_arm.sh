#!/bin/bash
# Amendment 5 primary arm: identical init/data/config; only ARM (a5_state | a5_ack) differs.  a5_arm.sh ARM SEED
set -euo pipefail
ARM=$1; SEED=$2; INIT=runs/rft_tiny_a5
O=runs/grpo_tiny_${ARM}_s${SEED}; L=logs/grpo_${ARM}_s${SEED}
[ -f $O/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.grpo --arm $ARM --model CohereLabs/tiny-aya-global \
  --init-adapter $INIT/final --task-jsonl data/bench/a5/a5_train.jsonl --out $O --steps 200 --num-generations 8 \
  --prompts-per-step 4 --max-steps-per-episode 18 --save-steps 25 --seed $SEED --use-vllm --rollout batched > $L.full.log 2>&1
cmd="slurm/a5_arm.sh $ARM $SEED"
.venv/bin/python scripts/a4_log_run.py --id A5_${ARM}_s$SEED --kind grpo --seed $SEED --cmd "$cmd" --adapter $O/final --data data/bench/a5/a5_train.jsonl > /dev/null
R=results/eval/a5/${ARM}_s${SEED}__a5_eval
[ -f $R.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval_a5 --model $INIT/merged --adapter $O/final \
  --tasks data/bench/a5/a5_eval.jsonl --max-turns 18 --out $R.jsonl > ${L}_eval.full.log 2>&1
.venv/bin/python scripts/a4_log_run.py --id A5_${ARM}_s${SEED}_eval --kind eval --cmd "run_eval_a5 a5_eval" --adapter $O/final --data data/bench/a5/a5_eval.jsonl --out $R.jsonl > /dev/null
python3 -c "import json;d=json.load(open('$R.summary.json'));a=d['a5']['all'];print('$ARM s$SEED', {k:round(a[k],3) for k in ('proof_valid','goal_final','duplicate','false_completion','safe_abstention','unnecessary_abstention','cert_valid','evidence_calls')}, d['ended_by'])"
