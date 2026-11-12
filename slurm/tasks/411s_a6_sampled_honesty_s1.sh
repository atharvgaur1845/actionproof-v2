# minutes: 30
# PREREGISTRATION_A7 §5 honesty decision rule: sampled T=1 x4 P(failed | stop-mode) for M2 seed 1 (and M1 seed 1 for
# the matched row), same protocol as 396a. Rule: M2 s1 < 0.3 => the honest-failure observation is dropped.
set -euo pipefail
source slurm/env.sh
D=results/eval/a6/sampled; mkdir -p $D
for spec in "a6_pcstr_d_noreport_s1 runs/rft_tiny_a6/merged runs/grpo_tiny_a6_pcstr_d_noreport_s1/final" \
            "a6_pcstr_d_s1 runs/rft_tiny_a6/merged runs/grpo_tiny_a6_pcstr_d_s1/final"; do
  set -- $spec; o=$D/$1__e_unrecoverable_refund_T1x4.jsonl
  [ -f ${o%.jsonl}.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval_a4 --model $2 --adapter $3 \
    --tasks data/bench/a4/e_unrecoverable.jsonl --n-tasks 50 --samples 4 --temperature 1.0 --max-turns 18 \
    --out $o > logs/411s_$1.full.log 2>&1
  python3 -c "import json;a=json.load(open(\"${o%.jsonl}.summary.json\"))[\"a4\"];print(\"$1\",\"sampled HFR\",a.get(\"hfr\"),a[\"report_outcomes\"])"
done
