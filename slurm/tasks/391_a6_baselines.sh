# minutes: 120
# after: 390_a6_init
# Amendment 6 baselines: warm start (neutral + prompt-only verify_a / verify_c), v2 init and v2 state arms on the new splits.
set -euo pipefail
source slurm/env.sh
bash slurm/a6_eval.sh init_a6 CohereLabs/tiny-aya-global runs/rft_tiny_a6/final neutral
for p in verify_a verify_c; do
  bash slurm/a6_eval.sh init_a6 CohereLabs/tiny-aya-global runs/rft_tiny_a6/final $p
  bash slurm/a6_eval.sh init_v2 CohereLabs/tiny-aya-global runs/rft_tiny_v2/final $p
done
for r in pcstr_d_s0 outcome_s0 ack_outcome_s0; do  # v2 models on the two new splits only
  for sp in e6_twin_rp e6_depth3; do
    o=results/eval/a6/v2_${r}__neutral__$sp.jsonl
    [ -f ${o%.jsonl}.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval_a4 --model runs/rft_tiny_v2/merged \
      --adapter runs/grpo_tiny_v2_$r/final --tasks data/bench/a6/$sp.jsonl --max-turns 18 --out $o > logs/391_${r}_$sp.full.log 2>&1
    python3 -c "import json;d=json.load(open('${o%.jsonl}.summary.json'));print('v2 $r $sp vsc',round(d['vsc'],3))"
  done
done
