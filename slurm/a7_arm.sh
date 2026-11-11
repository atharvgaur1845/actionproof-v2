#!/bin/bash
# Amendment 7 arm (PREREGISTRATION_A7 §1): Amendment 6 training config, pool a4_train_B, 18 turns; INIT is the warm
# start (rft_tiny_a7 for E7-3/E7-5, rft_tiny_a6 for E7-4). Then every A6 evaluation split.   a7_arm.sh ARM SEED INIT TAG
set -euo pipefail
ARM=$1; SEED=$2; INIT=runs/$3; TAG=$4; POOL=data/bench/a4/a4_train_B.jsonl
O=runs/grpo_tiny_${TAG}_${ARM}_s${SEED}; L=logs/grpo_${TAG}_${ARM}_s${SEED}
.venv/bin/python scripts/freeze_a7.py --check
cmd=".venv/bin/python -m actionproof.train.grpo --arm $ARM --model CohereLabs/tiny-aya-global --init-adapter $INIT/final \
 --task-jsonl $POOL --out $O --steps 200 --num-generations 8 --prompts-per-step 4 --max-steps-per-episode 18 \
 --save-steps 25 --seed $SEED --use-vllm --rollout batched"
[ -f $O/final/adapter_config.json ] || $cmd > $L.full.log 2>&1
.venv/bin/python scripts/a4_log_run.py --id A7_${TAG}_${ARM}_s$SEED --kind grpo --seed $SEED --cmd "$cmd" --adapter $O/final --data $POOL > /dev/null
python3 -c "import json;h=json.load(open('$O/log_history.json'));z=[x['frac_reward_zero_std'] for x in h if 'frac_reward_zero_std' in x];print('$TAG $ARM s$SEED zero_std first20',round(sum(z[:20])/20,3),'last20',round(sum(z[-20:])/20,3))"
bash slurm/a6_eval.sh ${TAG}_${ARM}_s${SEED} $INIT/merged $O/final
