# minutes: 30
# RFT v2 init (shared GRPO init) on the development test splits, greedy, neutral prompt: fills the init row of the arms table.
set -euo pipefail
source slurm/env.sh
for sp in test_iid test_ood_fault; do
  o=results/eval/r3/rft_tiny_v2__neutral__$sp.jsonl
  [ -f ${o%.jsonl}.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global \
      --adapter runs/rft_tiny_v2/final --split $sp --prompt neutral --out $o > logs/321_$sp.full.log 2>&1
  .venv/bin/python -m actionproof.eval.by_fault $o | tail -12
done
