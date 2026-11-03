# minutes: 90
# R1: DPO on first-divergence pairs from ON-POLICY RFT samples (curriculum sampling, T=1.0) + teacher
# samples, starting from the merged RFT init. Evaluate on val (neutral prompt).
set -euo pipefail
source slurm/env.sh
.venv/bin/python -m actionproof.train.dpo_data --samples results/curriculum/rft_tiny_train.jsonl results/teacher/full_32b_verify_a.jsonl \
  --out data/dpo/pairs_tiny.jsonl --max-per-task 3
[ -f runs/dpo_tiny/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.dpo --pairs data/dpo/pairs_tiny.jsonl \
  --model runs/rft_tiny_curated/merged --out runs/dpo_tiny --epochs 2 > logs/095_dpo.full.log 2>&1
grep -E "train_runtime" logs/095_dpo.full.log | tail -1 | cut -c1-160
.venv/bin/python -m actionproof.eval.run_eval --model runs/rft_tiny_curated/merged --adapter runs/dpo_tiny/final \
  --split val --prompt neutral --out results/eval/r1/dpo_tiny__neutral__val.jsonl > logs/095_eval.full.log 2>&1
python3 -c "import json;d=json.load(open('results/eval/r1/dpo_tiny__neutral__val.summary.json'));print('dpo', {k:round(d[k],3) for k in ('vsc','fcr','ivr','dsr','report_rate','verify_calls')})"
