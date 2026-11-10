# minutes: 180
# Amendment 6 warm start (PREREGISTRATION_A6 §3) + gate G-A6. Gate FAIL exits non-zero, so no arm (after: 390) runs.
set -euo pipefail
source slurm/env.sh
.venv/bin/python scripts/freeze_a6.py --check
.venv/bin/python -m pytest -q tests/test_a6.py
mkdir -p data/rft
[ -f data/rft/a6_demo_rows.jsonl ] || .venv/bin/python -m actionproof.train.a6_demos --out data/rft/a6_demo_rows.jsonl
.venv/bin/python -m actionproof.train.rft_data --teacher results/teacher/full_32b_verify_a.jsonl \
  --expert results/teacher/expert_refund_f050.jsonl --out data/rft/curated_v2_for_a6.jsonl > /dev/null
cat data/rft/curated_v2_for_a6.jsonl data/rft/a6_demo_rows.jsonl > data/rft/curated_a6.jsonl
wc -l data/rft/curated_v2_for_a6.jsonl data/rft/a6_demo_rows.jsonl data/rft/curated_a6.jsonl
[ -f runs/rft_tiny_a6/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft \
  --data data/rft/curated_a6.jsonl --out runs/rft_tiny_a6 --epochs 2 > logs/390_rft_a6.full.log 2>&1
.venv/bin/python scripts/a4_log_run.py --id A6_rft_init --kind rft --cmd "rft --data data/rft/curated_a6.jsonl --epochs 2" --adapter runs/rft_tiny_a6/final --data data/rft/curated_a6.jsonl > /dev/null
# Gate (same runner and settings as A4 X2; note run_eval_a4 lists the held-out tool, as X2 did).
G=results/eval/a6/gate_init__a4_train_B_T1x4; mkdir -p results/eval/a6
[ -f $G.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval_a4 --model CohereLabs/tiny-aya-global \
  --adapter runs/rft_tiny_a6/final --tasks data/bench/a4/a4_train_B.jsonl --n-tasks 100 --samples 4 --temperature 1.0 \
  --max-turns 18 --out $G.jsonl > logs/390_gate.full.log 2>&1
python3 - <<PY
import json,sys
a=json.load(open("$G.summary.json"))["a4"]; hfr=a.get("hfr",0.0)
print("G-A6", {"hfr_UR":round(hfr,3),"n_UR":a.get("class_UR",{}).get("n"),"pfr_RP":a.get("pfr"),
      "pwv_audit":a.get("pwv_add_audit_note",{}).get("rate"),"pwv_close":a.get("pwv_close_ticket",{}).get("rate"),
      "pwv_refund":a.get("pwv_issue_refund",{}).get("rate")})
ok=0.10<=hfr<=0.70; print("GATE", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PY
