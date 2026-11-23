# minutes: 60
# Algorithm discovery AD-D1 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL.md): ad_smoke.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29200
.venv/bin/python scripts/ad/freeze.py --check || exit 1
for est in awrb nostd; do
  OUT=runs/ad_smoke_$est; rm -rf $OUT
  AP_ROLLOUT_LOG=logs/ad_smoke_$est.rollouts.jsonl .venv/bin/python -m actionproof.ad.train --estimator $est \
    --pools data/bench/proc/cw_train.jsonl --seed 0 --steps 2 --init-adapter runs/rft_proc/final --out $OUT \
    --use-vllm > logs/ad_smoke_$est.log 2>&1 || { tail -40 logs/ad_smoke_$est.log; exit 1; }
  .venv/bin/python - "$est" <<'PY' || exit 1
import json, sys
est = sys.argv[1]
rows = [json.loads(l) for l in open(f"logs/ad_smoke_{est}.rollouts.jsonl")]
bad = [r["task_id"] for r in rows if r["ad"]["r_bar"] is None]
print(est, "episodes", len(rows), "no r_bar", len(bad), "mean reads", sum(r["n_verify"] for r in rows) / len(rows),
      "mean r_v", sum(r["score"]["r_v"] for r in rows) / len(rows), "mean r_bar",
      sum(r["ad"]["r_bar"] for r in rows) / len(rows))
assert not bad and len(rows) >= 64
cfg = json.load(open(f"runs/ad_smoke_{est}/run_config.json"))
assert cfg["scale_rewards"] == {"awrb": "group", "nostd": "none"}[est]
PY
  rm -rf $OUT
done
.venv/bin/python -m actionproof.ad.evaluate --model runs/rft_proc/merged --tag ad_smoke_warm --split cw_dev --n-tasks 10 \
  --out-dir logs/ad_smoke_eval > logs/ad_smoke_eval.log 2>&1 || { tail -30 logs/ad_smoke_eval.log; exit 1; }
tail -1 logs/ad_smoke_eval.log
echo SMOKE_OK
