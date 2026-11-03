# minutes: 90
# R1: RFT on the full verifier-curated 32B data (shared GRPO init) + the raw-teacher ablation
# (same episode count, no VSC filter). Evaluate both + base on val (neutral prompt, greedy).
set -euo pipefail
source slurm/env.sh
for v in curated raw; do
  [ -f runs/rft_tiny_$v/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft \
    --data data/rft/${v}_32b.jsonl --out runs/rft_tiny_$v --epochs 2 > logs/085_rft_$v.full.log 2>&1
  grep -E "train_runtime" logs/085_rft_$v.full.log | tail -1 | cut -c1-160
  .venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global --adapter runs/rft_tiny_$v/final \
    --split val --prompt neutral --out results/eval/r1/rft_tiny_${v}__neutral__val.jsonl > logs/085_eval_$v.full.log 2>&1
  python3 -c "import json;d=json.load(open('results/eval/r1/rft_tiny_${v}__neutral__val.summary.json'));print('$v', {k:round(d[k],3) for k in ('vsc','fcr','ivr','dsr','report_rate','verify_calls')})"
done
.venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global --split val --prompt neutral \
  --out results/eval/r1/base_tiny__neutral__val.jsonl > logs/085_eval_base.full.log 2>&1
python3 -c "import json;d=json.load(open('results/eval/r1/base_tiny__neutral__val.summary.json'));print('base', {k:round(d[k],3) for k in ('vsc','fcr','ivr','dsr','report_rate','verify_calls')})"
