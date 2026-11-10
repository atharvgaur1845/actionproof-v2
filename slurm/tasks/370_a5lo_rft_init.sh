# minutes: 150
# A5 re-run with headroom (D-A5-4): demos on 10% of each training family (was 50%, which put the init at 0.991
# proof-valid = ceiling). Everything else identical. Headroom gate: init greedy proof-valid on a5_val must be <= 0.80.
set -euo pipefail
source slurm/env.sh
D=results/teacher/a5_oracle_f010.jsonl
[ -f $D ] || .venv/bin/python -m actionproof.train.a5_demos --frac 0.1 --out $D
.venv/bin/python -m actionproof.train.rft_data --teacher results/teacher/full_32b_verify_a.jsonl \
  --expert results/teacher/expert_refund_f050.jsonl $D --out data/rft/curated_a5lo.jsonl | tr '\n' ' '; echo
[ -f runs/rft_tiny_a5lo/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft \
  --data data/rft/curated_a5lo.jsonl --out runs/rft_tiny_a5lo --epochs 2 > logs/370_rft.full.log 2>&1
for sp in a5_val a5_eval; do
  R=results/eval/a5/init_lo__neutral__none__$sp
  [ -f $R.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval_a5 --model CohereLabs/tiny-aya-global --adapter runs/rft_tiny_a5lo/final \
     --tasks data/bench/a5/$sp.jsonl --max-turns 18 --out $R.jsonl > logs/370_$sp.full.log 2>&1
  python3 -c "import json;a=json.load(open('$R.summary.json'))['a5']['all'];print('$sp',{k:round(a[k],3) for k in ('proof_valid','duplicate','false_completion','safe_abstention','unnecessary_abstention','cert_valid')})"
done
python3 -c "
import json,sys
pv=json.load(open('results/eval/a5/init_lo__neutral__none__a5_val.summary.json'))['a5']['all']['proof_valid']
print('HEADROOM GATE', pv, 'PASS' if pv<=0.80 else 'FAIL'); sys.exit(0 if pv<=0.80 else 1)"
