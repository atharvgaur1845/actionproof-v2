# minutes: 200
# R3: GRPO seed 0, arm=pcstr, Tiny Aya from the SHARED merged RFT init, learnable-band curriculum,
# batched vLLM rollouts, 200 steps x (4 prompts x 8). Resumes from the latest checkpoint if re-run.
# Then evaluate the final policy on val + test_iid + test_ood_fault (neutral prompt, greedy).
set -euo pipefail
source slurm/env.sh
O=runs/grpo_tiny_pcstr_s0
[ -f $O/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.grpo --arm pcstr --model CohereLabs/tiny-aya-global \
  --init-adapter runs/rft_tiny_curated/final --task-file data/curriculum/band_tiny.jsonl --out $O \
  --steps 200 --num-generations 8 --prompts-per-step 4 --save-steps 25 --seed 0 --use-vllm --rollout batched \
  > logs/100_grpo_pcstr.full.log 2>&1
tail -c 600 logs/100_grpo_pcstr.full.log | tr -d '\r' | tail -3
for s in val test_iid test_ood_fault; do
  .venv/bin/python -m actionproof.eval.run_eval --model runs/rft_tiny_curated/merged --adapter $O/final --split $s \
    --prompt neutral --out results/eval/r3/grpo_tiny_pcstr_s0__neutral__$s.jsonl > logs/100_eval_$s.full.log 2>&1
  python3 -c "import json;d=json.load(open('results/eval/r3/grpo_tiny_pcstr_s0__neutral__$s.summary.json'));print('pcstr', '$s', {k:round(d[k],3) for k in ('vsc','fcr','ivr','dsr','report_rate','verify_calls')})"
done
