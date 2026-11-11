# minutes: 60
# after: 397a_a7_forced_prefix 411_a7_pcstr_d_noreport_s0 412_a7_ack_outcome_s0 413_a7_pcstr_d_noFunder_s0
# Amendment 7 E7-1 extension: battery on the A7 init and arms (base rft_tiny_a7/merged) and on M2c (base rft_tiny_a6).
set -euo pipefail
source slurm/env.sh
.venv/bin/python scripts/freeze_a7.py --check
[ -d runs/rft_tiny_a7/merged ] || .venv/bin/python -c "from actionproof.train.merge import merge; merge('runs/rft_tiny_a7/final','CohereLabs/tiny-aya-global')"
.venv/bin/python -m actionproof.eval.forced_prefix score --prefixes data/bench/a7/forced_prefixes.jsonl \
  --base runs/rft_tiny_a6/merged --model a6_pcstr_d_noFunder_s0=runs/grpo_tiny_a6_pcstr_d_noFunder_s0/final \
  --out results/a7/forced_prefix.jsonl > logs/414_fp_a6.full.log 2>&1
.venv/bin/python -m actionproof.eval.forced_prefix score --prefixes data/bench/a7/forced_prefixes.jsonl \
  --base runs/rft_tiny_a7/merged --model a7_init= --model a7_pcstr_d_noreport_s0=runs/grpo_tiny_a7_pcstr_d_noreport_s0/final \
  --model a7_ack_outcome_s0=runs/grpo_tiny_a7_ack_outcome_s0/final --out results/a7/forced_prefix_a7.jsonl > logs/414_fp_a7.full.log 2>&1
cat results/a7/forced_prefix.jsonl results/a7/forced_prefix_a7.jsonl > results/a7/forced_prefix_all.jsonl
.venv/bin/python -m actionproof.eval.forced_prefix report results/a7/forced_prefix_all.jsonl | tee results/a7/forced_prefix.md
