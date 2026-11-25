# minutes: 240
# P4 (PREREGISTRATION_P4.md): arm Dw_s0 — SFT (actionproof.train.rft, 2 epochs, seed 0) then P4 evaluation.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p4.py --check || exit 1
mkdir -p logs results/p4
for sp in s4_api s4_wf4; do
  [ -f results/p4/Dw_s0__$sp.summary.json ] && { echo "skip $sp"; continue; }
  .venv/bin/python scripts/p4_eval.py --model runs/p1_dwarm/merged --tag Dw_s0 --split $sp --samples 4 > logs/p4_Dw_s0_$sp.log 2>&1 || { tail -30 logs/p4_Dw_s0_$sp.log; exit 1; }
  tail -1 logs/p4_Dw_s0_$sp.log
done
echo P4_Dw_s0_OK
