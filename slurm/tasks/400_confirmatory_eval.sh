# minutes: 120
# CONFIRMATORY (PREREGISTRATION §2-3). Run ONCE, after G3 (2026-11-11). Every final model, greedy, neutral prompt,
# on P = test_iid_c ∪ test_ood_fault_c. Do not move this file out of queue/*/hold before G3.
set -euo pipefail
source slurm/env.sh
[ "$(date +%Y%m%d)" -ge 20261111 ] || { echo "refusing: before G3"; exit 1; }
.venv/bin/python scripts/freeze_bench.py --check
D=results/eval/conf; mkdir -p $D
ev() {  # run base adapter split
  o=$D/${1}__neutral__$4.jsonl
  [ -f ${o%.jsonl}.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval --model $2 --adapter $3 --split $4 \
      --prompt neutral --out $o > logs/400_${1}_$4.full.log 2>&1
  python3 -c "import json;d=json.load(open('${o%.jsonl}.summary.json'));print('$1 $4', {k:round(d[k],3) for k in ('vsc','fcr','ivr','dsr','verify_calls')})"
}
for sp in test_iid_c test_ood_fault_c; do
  ev rft_tiny_v2 CohereLabs/tiny-aya-global runs/rft_tiny_v2/final $sp
  for r in pcstr_d_s0 pcstr_d_s1 outcome_s0 outcome_s1 tool_trust_s0 tool_trust_s1 pcstr_s0 pcstr_c_s0 llm_judge_s0; do
    ev grpo_tiny_v2_$r runs/rft_tiny_v2/merged runs/grpo_tiny_v2_$r/final $sp
  done
done
.venv/bin/python -m actionproof.eval.confirmatory --dir $D --out results/confirmatory.json
