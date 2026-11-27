# minutes: 50
# RG1 development pilot (txn_dev only; no sealed split): D seed 0 with and without the Recovery Gate, and the base
# model with the gate. Used to find gate bugs and unclear gate messages before the registration freeze.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/rg1
.venv/bin/python -m pytest -q tests/recovery_gate > logs/rg1_tests.log 2>&1 || { tail -30 logs/rg1_tests.log; exit 1; }
tail -1 logs/rg1_tests.log
run() { tag=$1; shift; [ -f results/rg1/${tag}__txn_dev.summary.json ] && { echo "skip $tag"; return 0; }
  rm -f results/rg1/${tag}__txn_dev.jsonl
  .venv/bin/python scripts/rg1_eval.py --tag $tag --split txn_dev --samples 2 "$@" > logs/rg1_$tag.log 2>&1 || { tail -40 logs/rg1_$tag.log; return 1; }
  tail -1 logs/rg1_$tag.log; }
run dev_D_s0_gate --model runs/p1_dwarm/merged --adapter runs/p1_d_s0/final --rules full || exit 1
run dev_D_s0_none --model runs/p1_dwarm/merged --adapter runs/p1_d_s0/final --rules none || exit 1
run dev_base_gate --model CohereLabs/tiny-aya-global --rules full || exit 1
echo RG1_DEV_PILOT_OK
