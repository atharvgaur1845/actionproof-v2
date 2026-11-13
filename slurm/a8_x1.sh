#!/bin/bash
# Amendment 8 X1 (PREREGISTRATION_A8): the A7 ack_outcome run (task 412) with ONE change: the task list drops the
# refund-timeout and refund-UR tasks of pool B (data/bench/a8/x1_task_ids.jsonl, 340 of 400). Same warm start, reward,
# optimiser, steps, seed, turns. Training rollouts are kept (AP_ROLLOUT_LOG). Then every A6 evaluation split.
#
# Time-sliced: the task trains until 15 min before this worker's wall time, then re-queues itself; the next worker
# resumes from the last checkpoint (grpo.py resumes when checkpoint-* exists). Evaluation splits are skipped once
# their summary exists, so evaluation resumes the same way.     a8_x1.sh <path of the task script, for re-queueing>
set -euo pipefail
SELF=$1; ROOT=/scratch/$USER/actionproof
ARM=ack_outcome; SEED=0; INIT=runs/rft_tiny_a7; POOL=data/bench/a4/a4_train_B.jsonl; KEEP=data/bench/a8/x1_task_ids.jsonl
O=runs/grpo_tiny_a8x1_ack_outcome_s0; L=logs/grpo_a8x1_ack_outcome_s0; mkdir -p $O
.venv/bin/python scripts/freeze_a8.py --check
left() { squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{ if (NF==4) print $1*1440+$2*60+$3; else if (NF==3) print $1*60+$2; else print $1 }'; }
requeue() { echo "$(date +%T) time slice ended ($1); re-queueing $(basename "$SELF")"; cp "$SELF" "$ROOT/queue/h200/pending/"; exit 0; }
export AP_ROLLOUT_LOG=$O/train_rollouts.jsonl  # one line per training episode, with the trainer step
cmd=".venv/bin/python -m actionproof.train.grpo --arm $ARM --model CohereLabs/tiny-aya-global --init-adapter $INIT/final \
 --task-jsonl $POOL --task-file $KEEP --out $O --steps 200 --num-generations 8 --prompts-per-step 4 \
 --max-steps-per-episode 18 --save-steps 25 --seed $SEED --use-vllm --rollout batched"
if [ ! -f $O/final/adapter_config.json ]; then
  last=$(ls -d $O/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)  # a checkpoint cut mid-write cannot resume
  [ -n "$last" ] && [ ! -f "$last/trainer_state.json" ] && { echo "drop incomplete $last"; rm -rf "$last"; }
  budget=$(( $(left) - 15 )); [ $budget -lt 20 ] && requeue "train, ${budget}m left"
  echo "$(date +%T) train slice ${budget}m"
  timeout --kill-after=60 ${budget}m $cmd >> $L.full.log 2>&1 || { rc=$?; [ $rc = 124 ] && requeue train; exit $rc; }
  .venv/bin/python scripts/a4_log_run.py --id A8_x1_${ARM}_s$SEED --kind grpo --seed $SEED --cmd "$cmd" --adapter $O/final \
    --data $KEEP > /dev/null
  python3 -c "import json;h=json.load(open('$O/log_history.json'));z=[x['frac_reward_zero_std'] for x in h if 'frac_reward_zero_std' in x];print('a8x1 zero_std first20',round(sum(z[:20])/20,3),'last20',round(sum(z[-20:])/20,3))"
fi
budget=$(( $(left) - 15 )); [ $budget -lt 25 ] && requeue "eval, ${budget}m left"
timeout --kill-after=60 ${budget}m bash slurm/a6_eval.sh a8x1_${ARM}_s${SEED} $INIT/merged $O/final || { rc=$?; [ $rc = 124 ] && requeue eval; exit $rc; }
echo "X1 done"
