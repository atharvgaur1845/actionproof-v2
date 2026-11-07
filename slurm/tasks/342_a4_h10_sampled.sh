# minutes: 40
# after: 341_a4_h10_ack_outcome_s1
# Amendment 4 X4 / H10: sampled post-write verification of ack_outcome (same protocol as X1: val, T=1 x 8, <= 10 turns).
set -euo pipefail
source slurm/env.sh
D=results/eval/a4/h11
for s in 0 1; do
  o=$D/ack_outcome_s${s}__val_T1x8.jsonl
  cmd=".venv/bin/python -m actionproof.eval.run_eval --model runs/rft_tiny_v2/merged --adapter runs/grpo_tiny_v2_ack_outcome_s$s/final --split val --prompt neutral --samples 8 --temperature 1.0 --max-turns 10 --seed 0 --out $o"
  [ -f ${o%.jsonl}.summary.json ] || $cmd > logs/342_s$s.full.log 2>&1
  .venv/bin/python scripts/a4_log_run.py --id X4_ack_outcome_s$s --kind eval_sampled --cmd "$cmd" --adapter runs/grpo_tiny_v2_ack_outcome_s$s/final --out $o > /dev/null
  .venv/bin/python -m actionproof.eval.a4_metrics $o | cut -c1-400
done
