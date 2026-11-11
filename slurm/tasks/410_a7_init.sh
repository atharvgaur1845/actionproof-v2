# minutes: 200
# Amendment 7 E7-2 (PREREGISTRATION_A7 §3): warm start rft_tiny_a7 = A6 SFT rows + 20 credit demonstrations, then
# gate G-A7. Gate FAIL exits non-zero, so no E7-3 arm (after: 410) runs.
set -euo pipefail
source slurm/env.sh
.venv/bin/python scripts/freeze_a7.py --check
.venv/bin/python -m pytest -q tests/test_a7.py
[ -f data/rft/a7_credit_demo_rows.jsonl ] || .venv/bin/python -m actionproof.train.a7_demos --out data/rft/a7_credit_demo_rows.jsonl
cat data/rft/curated_a6.jsonl data/rft/a7_credit_demo_rows.jsonl > data/rft/curated_a7.jsonl
wc -l data/rft/curated_a6.jsonl data/rft/a7_credit_demo_rows.jsonl data/rft/curated_a7.jsonl
[ -f runs/rft_tiny_a7/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft \
  --data data/rft/curated_a7.jsonl --out runs/rft_tiny_a7 --epochs 2 > logs/410_rft_a7.full.log 2>&1
.venv/bin/python scripts/a4_log_run.py --id A7_rft_init --kind rft --cmd "rft --data data/rft/curated_a7.jsonl --epochs 2" --adapter runs/rft_tiny_a7/final --data data/rft/curated_a7.jsonl > /dev/null
D=results/eval/a7; mkdir -p $D
G=$D/gate_init__a4_train_B_T1x4
[ -f $G.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval_a4 --model CohereLabs/tiny-aya-global \
  --adapter runs/rft_tiny_a7/final --tasks data/bench/a4/a4_train_B.jsonl --n-tasks 100 --samples 4 --temperature 1.0 \
  --max-turns 18 --out $G.jsonl > logs/410_gate.full.log 2>&1
H=$D/a7_init__neutral__e_heldout
[ -f $H.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval_a4 --model CohereLabs/tiny-aya-global \
  --adapter runs/rft_tiny_a7/final --tasks data/bench/a4/e_heldout.jsonl --max-turns 18 --out $H.jsonl > logs/410_heldout.full.log 2>&1
python3 - <<PY
import json,sys
a=json.load(open("$G.summary.json"))["a4"]; hfr=a.get("hfr",0.0)
ok_c=0; n=0
for l in open("$H.jsonl"):
    r=json.loads(l); n+=1
    arg=lambda s: s.get("args") if isinstance(s.get("args"),dict) else {}
    t=[s for s in r["trace"] if not s.get("by")]
    rk=next((arg(s).get("idempotency_key") for s in t if s["tool"]=="issue_refund"),None)
    ok_c+=any(s["tool"]=="apply_store_credit" and arg(s).get("idempotency_key")!=rk for s in t)
rate=ok_c/n
print("G-A7", {"hfr_UR":round(hfr,3),"pwv_refund":a.get("pwv_issue_refund",{}).get("rate"),"noncolliding_credit_rate":round(rate,3)})
ok=0.10<=hfr<=0.70 and rate>=0.30; print("GATE", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY
