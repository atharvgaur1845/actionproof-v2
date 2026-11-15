# minutes: 80
# Amendment 10 addendum: existing v2-round policies in ActionProof-Bench v2 (test_iid_v2, 234 tasks), greedy, 12 turns.
set -uo pipefail
source slurm/env.sh
.venv/bin/python scripts/freeze_a10b.py --check || exit 1
mkdir -p results/eval/v2env
for m in "rft_tiny_v2:" "pcstr_d_s0:runs/grpo_tiny_v2_pcstr_d_s0/final" "ack_outcome_s0:runs/grpo_tiny_v2_ack_outcome_s0/final" \
         "outcome_s0:runs/grpo_tiny_v2_outcome_s0/final" "tool_trust_s0:runs/grpo_tiny_v2_tool_trust_s0/final"; do
  tag=${m%%:*}; ad=${m#*:}
  o=results/eval/v2env/${tag}__neutral__test_iid_v2.jsonl
  [ -f ${o%.jsonl}.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval_a4 --model runs/rft_tiny_v2/merged \
    ${ad:+--adapter $ad} --tasks data/bench/v2/test_iid_v2.jsonl --prompt neutral --max-turns 12 --out $o > logs/422_$tag.full.log 2>&1
  python3 -c "import json;d=json.load(open('${o%.jsonl}.summary.json'));print('$tag', round(d['vsc'],3))"
done
