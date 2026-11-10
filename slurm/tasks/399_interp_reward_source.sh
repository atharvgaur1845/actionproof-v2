# minutes: 150
# after: 395_a6_ack_outcome_s0
# Mechanistic analysis of the reward-source effect (docs/INTERP_PLAN.md). Existing checkpoints only, no training.
set -euo pipefail
source slurm/env.sh
O=results/interp/rs
[ -f $O/prefixes.jsonl ] || python3 -m actionproof.interp.rs_prefixes --v2 results/eval/a4/h11/*__val_T1x8.jsonl \
  $(ls results/eval/r3/*__neutral__test_iid.jsonl results/eval/r3/*__neutral__test_ood_fault.jsonl | grep -E "rft_tiny_v2__|ack_outcome|v2_outcome_s|pcstr_d_s|v2_tool_trust_s") \
  --a5 results/eval/a5/init__neutral__none__a5_eval.jsonl --out $O/prefixes.jsonl
declare -A M=([init]=runs/rft_tiny_v2/merged [ack_outcome_s0]=runs/grpo_tiny_v2_ack_outcome_s0/merged
  [ack_outcome_s1]=runs/grpo_tiny_v2_ack_outcome_s1/merged [outcome_s0]=runs/grpo_tiny_v2_outcome_s0/merged
  [outcome_s1]=runs/grpo_tiny_v2_outcome_s1/merged [pcstr_d_s0]=runs/grpo_tiny_v2_pcstr_d_s0/merged
  [tool_trust_s0]=runs/grpo_tiny_v2_tool_trust_s0/merged)
for t in init ack_outcome_s0 ack_outcome_s1 outcome_s0 outcome_s1 pcstr_d_s0 tool_trust_s0; do
  [ -f $O/${t}__A1_A2.npz ] || .venv/bin/python -m actionproof.interp.rs_extract --prefixes $O/prefixes.jsonl --kinds A1 A2 --model ${M[$t]} --tag $t --out $O
done
for k in M2 CD; do
  [ -f $O/a5_init__$k.npz ] || .venv/bin/python -m actionproof.interp.rs_extract --prefixes $O/prefixes.jsonl --kinds $k --model runs/rft_tiny_a5/merged --tag a5_init --out $O --no-tools
done
.venv/bin/python -m actionproof.interp.rs_analyze --tags init ack_outcome_s0 ack_outcome_s1 outcome_s0 outcome_s1 pcstr_d_s0 tool_trust_s0 --out $O/analysis.json
.venv/bin/python -m actionproof.interp.rs_intervene --ablate-tag outcome_s0 --ablate-model runs/grpo_tiny_v2_outcome_s0/merged \
  --steer-tag ack_outcome_s0 --steer-model runs/grpo_tiny_v2_ack_outcome_s0/merged --out $O/m3.json | tail -40
