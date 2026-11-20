# minutes: 40
# Amendment 14 pilot (b) (PREREGISTRATION_A14 §6): one checkpoint not analysed in A14 (R_V seed 0, step 50) on 10
# proc_dev tasks x 5 branches, to check rendering, parsing, recording and time per episode. Metrics are not read.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/a14/pilot
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/a14_merge.log 2>&1 || { tail -20 logs/a14_merge.log; exit 1; }
.venv/bin/python scripts/cef_eval.py model --model runs/rft_proc/merged --adapter runs/a12_proc_rv_s0/checkpoint-50 \
  --tag pilot_rv_s0_step50 --split proc_dev --n-tasks 10 --out-dir results/a14/pilot > logs/a14_pilot.full.log 2>&1 \
  || { tail -30 logs/a14_pilot.full.log; exit 1; }
python3 - <<'PY'
import json, collections
rs = [json.loads(l) for l in open("results/a14/pilot/pilot_rv_s0_step50__proc_dev__agent.jsonl")]
print("episodes", len(rs), "ended_by", dict(collections.Counter(r["ended_by"] for r in rs)),
      "invalid_calls", sum(r["invalid"] for r in rs), "first=invalid/none",
      sum(r["first"]["action"] in ("invalid", "none") for r in rs))
print("seconds", json.load(open("results/a14/pilot/summary__pilot_rv_s0_step50__proc_dev__agent.json"))["_meta"]["seconds"])
PY
