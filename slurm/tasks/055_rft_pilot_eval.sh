# minutes: 30
# Evaluate the RFT pilot adapter (merged; vLLM LoRA breaks on Cohere tied embeddings) on the same
# 60 test_iid tasks as the base-model eval pilot, NEUTRAL prompt.
set -euo pipefail
source slurm/env.sh
.venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global --adapter runs/rft_pilot_tiny/final \
  --split test_iid --n-tasks 60 --prompt neutral --out results/eval/pilot_rft_tiny_neutral.jsonl > logs/055_rft_pilot_eval.full.log 2>&1
tail -14 logs/055_rft_pilot_eval.full.log
