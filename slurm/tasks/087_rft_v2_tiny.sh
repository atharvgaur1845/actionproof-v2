# minutes: 80
# RFT v2 (deviation 2026-11-04): verifier-curated 32B data + lean scripted-expert demos on 50% of
# refund train tasks. The teacher never verifies after a success ack (0/560 on F1), so v1 data had
# no F1/F3 recoveries and GRPO got no gradient there. frac 0.5 keeps post-write verification
# stochastic in the student (learnable band). Eval on val with per-fault VSC.
set -euo pipefail
source slurm/env.sh
E=results/teacher/expert_refund_f050.jsonl
[ -f $E ] || .venv/bin/python -m actionproof.train.expert_demos --split train --frac 0.5 --out $E
.venv/bin/python -m actionproof.train.rft_data --teacher results/teacher/full_32b_verify_a.jsonl --expert $E \
  --out data/rft/curated_v2_32b.jsonl | tr '\n' ' '; echo
[ -f runs/rft_tiny_v2/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft \
  --data data/rft/curated_v2_32b.jsonl --out runs/rft_tiny_v2 --epochs 2 > logs/087_rft_v2.full.log 2>&1
R=results/eval/r1/rft_tiny_v2__neutral__val
[ -f $R.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global \
  --adapter runs/rft_tiny_v2/final --split val --prompt neutral --out $R.jsonl > logs/087_eval_v2.full.log 2>&1
.venv/bin/python -m actionproof.eval.by_fault $R.jsonl results/eval/r1/rft_tiny_curated__neutral__val.jsonl
