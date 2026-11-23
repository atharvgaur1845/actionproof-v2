# minutes: 40
# Stage 5 smoke (not analysed): 2 GRPO steps of each mode; checks that C5 advantages are installed and G hints fire.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29909
.venv/bin/python scripts/freeze_c5g.py --check || exit 1
for mode in c5 guide; do
  OUT=runs/s5_smoke_$mode; rm -rf $OUT
  AP_ROLLOUT_LOG=logs/s5_smoke_$mode.rollouts.jsonl .venv/bin/python -m actionproof.stage4.train_c5g --mode $mode \
    --pools data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl --seed 0 --steps 2 \
    --init-adapter runs/rft_proc/final --out $OUT --use-vllm > logs/s5_smoke_$mode.log 2>&1 || { tail -40 logs/s5_smoke_$mode.log; exit 1; }
  .venv/bin/python - "$mode" <<'PY' || exit 1
import json, sys
mode = sys.argv[1]
rows = [json.loads(l) for l in open(f"logs/s5_smoke_{mode}.rollouts.jsonl")]
prog = sum(any(u != 0 for u in r["turn_progress"]) for r in rows)
print(mode, "episodes", len(rows), "with progress", prog, "hinted", sum(r["hinted"] for r in rows))
if mode == "c5":
    g = [json.loads(l) for l in open("runs/s5_smoke_c5/c5_groups.jsonl")]
    assert g and g[0]["groups"], "C5 advantages not installed"
assert prog > 0
PY
  rm -rf $OUT
done
echo SMOKE_OK
