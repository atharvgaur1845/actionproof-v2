# minutes: 30
# after: 330_a4_integrity
# Amendment 4 X2 / gate G-A4: init on the regime-B TRAINING pool (training env, H absent), T=1 x 4, 100 tasks, 18 turns.
# Gate: expected verifying rollouts per GRPO step after acked add_audit_note/close_ticket >= 0.5 (rate >= 1.6%).
set -euo pipefail
source slurm/env.sh
o=results/eval/a4/gate/init__a4_train_B_T1x4.jsonl; mkdir -p $(dirname $o)
cmd=".venv/bin/python -m actionproof.eval.run_eval_a4 --model CohereLabs/tiny-aya-global --adapter runs/rft_tiny_v2/final --tasks data/bench/a4/a4_train_B.jsonl --n-tasks 100 --samples 4 --temperature 1.0 --max-turns 18 --seed 0 --out $o"
[ -f ${o%.jsonl}.summary.json ] || $cmd > logs/332_gate.full.log 2>&1
.venv/bin/python scripts/a4_log_run.py --id X2_gate --kind eval_sampled --cmd "$cmd" --adapter runs/rft_tiny_v2/final --data data/bench/a4/a4_train_B.jsonl --out $o > /dev/null
.venv/bin/python - $o <<'PY'
import json, sys
from actionproof.eval.a4_metrics import pwv
R = [json.loads(l) for l in open(sys.argv[1])]
for tool in ("issue_refund", "add_audit_note", "close_ticket"):
    v = [x for x in (pwv(r["trace"], tool) for r in R) if x is not None]
    print(tool, "PWV", round(sum(v) / max(1, len(v)), 4), "n_acked_eps", len(v))
v = [x for r in R for x in (pwv(r["trace"], t) for t in ("add_audit_note", "close_ticket")) if x is not None]
rate = sum(v) / max(1, len(v))
print("GATE non-refund PWV rate", round(rate, 4), "expected verifying rollouts/step", round(rate * 32, 2),
      "->", "PASS" if rate * 32 >= 0.5 else "FAIL (H12/H13 predicted null for exploration reasons)")
PY
