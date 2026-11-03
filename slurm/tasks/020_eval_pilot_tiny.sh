# minutes: 45
# R0 pilot: Tiny Aya Global on 60 test_iid tasks, neutral vs verify_a prompt (greedy).
set -euo pipefail
source slurm/env.sh
for p in neutral verify_a; do
  .venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global --split test_iid \
    --n-tasks 60 --prompt $p --out results/eval/pilot_tiny_global_${p}.jsonl > logs/$(basename $0 .sh)_${p:-x}.full.log 2>&1; tail -25 logs/$(basename $0 .sh)_${p:-x}.full.log | grep -v -i warn
done
