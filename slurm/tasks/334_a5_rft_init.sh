# minutes: 120
# after: 333_gates
# Amendment 5 shared init: v2 RFT rows (rebuilt identically: same teacher file, v2 expert demos, seed 0) + A5 oracle
# demos on 50% of each training family (held-out families absent). Then init baselines on a5_eval.
set -euo pipefail
source slurm/env.sh
D=results/teacher/a5_oracle_f050.jsonl
[ -f $D ] || .venv/bin/python -m actionproof.train.a5_demos --frac 0.5 --out $D
.venv/bin/python -m actionproof.train.rft_data --teacher results/teacher/full_32b_verify_a.jsonl \
  --expert results/teacher/expert_refund_f050.jsonl $D --out data/rft/curated_a5.jsonl | tr '\n' ' '; echo
[ -f runs/rft_tiny_a5/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft \
  --data data/rft/curated_a5.jsonl --out runs/rft_tiny_a5 --epochs 2 > logs/360_rft_a5.full.log 2>&1
.venv/bin/python scripts/a4_log_run.py --id A5_rft_init --kind rft --cmd "rft --data data/rft/curated_a5.jsonl --epochs 2" --adapter runs/rft_tiny_a5/final --data data/rft/curated_a5.jsonl > /dev/null
for cfg in "neutral:none" "neutral:raw" "neutral:vbr" "neutral:raw+vbr" "evidence_contract:none"; do
  p=${cfg%%:*}; m=${cfg#*:}; R=results/eval/a5/init__${p}__${m}__a5_eval
  [ -f $R.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval_a5 --model CohereLabs/tiny-aya-global --adapter runs/rft_tiny_a5/final \
     --tasks data/bench/a5/a5_eval.jsonl --max-turns 18 --prompt $p --middleware $m --out $R.jsonl > logs/360_init_${p}_${m}.full.log 2>&1
  python3 -c "import json;d=json.load(open('$R.summary.json'));a=d['a5']['all'];print('init $p $m', {k:round(a[k],3) for k in ('proof_valid','duplicate','false_completion','safe_abstention','unnecessary_abstention','cert_valid','evidence_calls','harness_calls')})"
done
