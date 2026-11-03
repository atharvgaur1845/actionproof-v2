# minutes: 90
# R1 data: Aya Expanse 32B teacher, verify_a prompt, all 400 train tasks x 8 samples (T=0.7),
# then curated + raw RFT data (per-task cap 2) and DPO pairs. ~38% keep-rate in the pilot.
set -euo pipefail
source slurm/env.sh
L=logs/070_teacher_full.full.log
.venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/aya-expanse-32b --split train \
  --prompt verify_a --samples 8 --temperature 0.7 --gpu-mem 0.88 \
  --out results/teacher/full_32b_verify_a.jsonl > $L 2>&1; tail -14 $L
T=results/teacher/full_32b_verify_a.jsonl
.venv/bin/python -m actionproof.train.rft_data --teacher $T --out data/rft/curated_32b.jsonl
.venv/bin/python -m actionproof.train.rft_data --teacher $T --raw --match-curated data/rft/curated_32b.jsonl --out data/rft/raw_32b.jsonl
.venv/bin/python -m actionproof.train.dpo_data --samples $T --out data/dpo/teacher_pairs.jsonl
