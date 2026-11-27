# minutes: 50
# RG1 development pilot 2 (no sealed split): the gate revised after pilot 1 (read first then decide; concrete report
# rejections; stable key after a non-ok reply). D seed 0 with the gate on txn_dev and txn_val; without it on txn_val.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/rg1
.venv/bin/python -m pytest -q tests/recovery_gate > logs/rg1_tests2.log 2>&1 || { tail -30 logs/rg1_tests2.log; exit 1; }
tail -1 logs/rg1_tests2.log
run() { tag=$1; sp=$2; shift 2; [ -f results/rg1/${tag}__$sp.summary.json ] && { echo "skip $tag"; return 0; }
  rm -f results/rg1/${tag}__$sp.jsonl
  .venv/bin/python scripts/rg1_eval.py --tag $tag --split $sp --samples 2 "$@" > logs/rg1_${tag}_$sp.log 2>&1 || { tail -40 logs/rg1_${tag}_$sp.log; return 1; }
  tail -1 logs/rg1_${tag}_$sp.log; }
M="--model runs/p1_dwarm/merged --adapter runs/p1_d_s0/final"
run dev2_D_s0_gate txn_dev $M --rules full || exit 1
run val_D_s0_gate txn_val $M --rules full || exit 1
run val_D_s0_none txn_val $M --rules none || exit 1
echo RG1_DEV_PILOT2_OK
