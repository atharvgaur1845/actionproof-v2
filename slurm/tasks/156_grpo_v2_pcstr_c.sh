# minutes: 200
# after: 091_curriculum_v2
# R3 round v2 / H8: GRPO seed 0, arm=pcstr_c (PCSTR-d + 0.02 per verification call), shared RFT v2 init.
set -euo pipefail
source slurm/env.sh
slurm/grpo_arm.sh pcstr_c runs/rft_tiny_v2 data/curriculum/band_tiny_v2.jsonl v2
R=results/eval/r3/grpo_tiny_v2_pcstr_c_s0__neutral__test_calib_swap
[ -f $R.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval --model runs/rft_tiny_v2/merged \
  --adapter runs/grpo_tiny_v2_pcstr_c_s0/final --split test_calib_swap --prompt neutral --out $R.jsonl > logs/156_calib_swap.full.log 2>&1
.venv/bin/python -m actionproof.eval.calibration results/eval/r3/grpo_tiny_v2_pcstr_c_s0__neutral__test_iid.jsonl $R.jsonl
