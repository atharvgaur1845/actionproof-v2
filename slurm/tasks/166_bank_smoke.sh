# minutes: 20
# Domain B pipeline smoke: BASE Tiny Aya only, 10 tasks of test_ood_domain (disclosed in PREREGISTRATION
# amendment 1). Checks prompt adaptation, tool parsing and the bank verifier end to end on GPU.
set -euo pipefail
source slurm/env.sh
R=results/eval/smoke/base_tiny__neutral__test_ood_domain_n10
mkdir -p results/eval/smoke
.venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global --split test_ood_domain --n-tasks 10 \
  --prompt neutral --out $R.jsonl > logs/166_bank_smoke.full.log 2>&1
python3 -c "import json;d=json.load(open('$R.summary.json'));print({k:d[k] for k in ('vsc','fcr','report_rate','verify_calls','mean_turns')})"
python3 -c "
import json
for l in list(open('$R.jsonl'))[:2]:
    e=json.loads(l); print(e['task_id'], e['ended_by'], [s['tool'] for s in e['trace']], e['verdict']['reported'])"
