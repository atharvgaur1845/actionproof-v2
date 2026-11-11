# minutes: 120
# after: 396q_a6_pcstr_d_noreport_s1
# Amendment 7 E7-1 (PREREGISTRATION_A7 §4): forced-prefix battery on every A6 arm, both schemas. Inference only.
set -euo pipefail
source slurm/env.sh
.venv/bin/python scripts/freeze_a7.py --check
R=runs; M=()
M+=(--model init=)
for a in pcstr_d_s0 pcstr_d_s1 pcstr_d_noreport_s0 pcstr_d_noreport_s1 pcstr_d_noND_s0 outcome_s0 ack_outcome_s0; do
  M+=(--model a6_$a=$R/grpo_tiny_a6_$a/final); done
for a in pcstr_d_s0 pcstr_d_noreport_s0 pcstr_d_noND_s0; do for c in 50 100 150; do
  M+=(--model a6_${a}@$c=$R/grpo_tiny_a6_$a/checkpoint-$c); done; done
mkdir -p results/a7
.venv/bin/python -m actionproof.eval.forced_prefix score --prefixes data/bench/a7/forced_prefixes.jsonl \
  --base $R/rft_tiny_a6/merged "${M[@]}" --out results/a7/forced_prefix.jsonl > logs/397a_forced_prefix.full.log 2>&1
.venv/bin/python -m actionproof.eval.forced_prefix report results/a7/forced_prefix.jsonl | tee results/a7/forced_prefix.md
