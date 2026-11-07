# minutes: 60
# after: 330_a4_integrity
# Amendment 4 X1 / H11: sampled post-write verification, T=1 x 8 on val, <= 10 turns (registered budget).
set -euo pipefail
source slurm/env.sh
D=results/eval/a4/h11; mkdir -p $D
ev() {  # tag base adapter
  o=$D/$1__val_T1x8.jsonl
  cmd=".venv/bin/python -m actionproof.eval.run_eval --model $2 --adapter $3 --split val --prompt neutral --samples 8 --temperature 1.0 --max-turns 10 --seed 0 --out $o"
  [ -f ${o%.jsonl}.summary.json ] || $cmd > logs/331_$1.full.log 2>&1
  .venv/bin/python scripts/a4_log_run.py --id X1_$1 --kind eval_sampled --cmd "$cmd" --adapter $3 --out $o > /dev/null
  .venv/bin/python -m actionproof.eval.a4_metrics $o | python3 -c "import json,sys;l=sys.stdin.read();n,j=l.split(' ',1);d=json.loads(j);print('$1', {k:d[k] for k in d if k.startswith('pwv_') or k in ('vsc','report_outcomes')})"
}
ev init CohereLabs/tiny-aya-global runs/rft_tiny_v2/final
for r in tool_trust_s0 outcome_s0 pcstr_d_s0; do ev $r runs/rft_tiny_v2/merged runs/grpo_tiny_v2_$r/final; done
