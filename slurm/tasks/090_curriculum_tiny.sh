# minutes: 40
# Learnable-band curriculum: sample the RFT init 8x per train task (T=1.0), keep 10-90% VSC tasks.
set -euo pipefail
source slurm/env.sh
.venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global --adapter runs/rft_tiny_curated/final \
  --split train --prompt neutral --samples 8 --temperature 1.0 --out results/curriculum/rft_tiny_train.jsonl > logs/090_curriculum.full.log 2>&1
.venv/bin/python -m actionproof.train.curriculum --samples results/curriculum/rft_tiny_train.jsonl --out data/curriculum/band_tiny.jsonl
