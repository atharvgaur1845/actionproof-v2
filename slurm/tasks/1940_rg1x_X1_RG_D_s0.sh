# minutes: 40
# RG1 addendum 1 (PREREGISTRATION_RG1_ADDENDUM1.md, descriptive): X1_RG_D_s0 on sel1_test, the frozen gate.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/rg1
[ -f results/rg1/X1_RG_D_s0__sel1_test.summary.json ] && { echo done already; exit 0; }
rm -f results/rg1/X1_RG_D_s0__sel1_test.jsonl
.venv/bin/python scripts/rg1x_sel_eval.py --model runs/p1_dwarm/merged --adapter runs/p1_d_s0/final --tag X1_RG_D_s0 > logs/rg1x_X1_RG_D_s0.log 2>&1 || { tail -40 logs/rg1x_X1_RG_D_s0.log; exit 1; }
tail -1 logs/rg1x_X1_RG_D_s0.log
echo RG1X_X1_RG_D_s0_OK
