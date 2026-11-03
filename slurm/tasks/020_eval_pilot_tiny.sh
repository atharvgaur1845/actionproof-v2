# minutes: 45
# R0 pilot: Tiny Aya Global on 60 test_iid tasks, neutral vs verify_a prompt (greedy).
set -euo pipefail
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
for p in neutral verify_a; do
  .venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global --split test_iid \
    --n-tasks 60 --prompt $p --out results/eval/pilot_tiny_global_${p}.jsonl 2>&1 | grep -v -i warn | tail -15
done
