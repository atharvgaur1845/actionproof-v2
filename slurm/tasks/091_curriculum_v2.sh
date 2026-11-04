# minutes: 40
# after: 087_rft_v2_tiny
# Learnable band for the v2 init: sample RFT v2 8x per train task (T=1.0), keep 10-90% VSC tasks.
set -euo pipefail
source slurm/env.sh
S=results/curriculum/rft_tiny_v2_train.jsonl
[ -f ${S%.jsonl}.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global \
  --adapter runs/rft_tiny_v2/final --split train --prompt neutral --samples 8 --temperature 1.0 --out $S > logs/091_curriculum.full.log 2>&1
.venv/bin/python -m actionproof.train.curriculum --samples $S --out data/curriculum/band_tiny_v2.jsonl
.venv/bin/python -m actionproof.eval.by_fault $S
