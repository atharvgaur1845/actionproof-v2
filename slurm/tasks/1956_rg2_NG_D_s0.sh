# minutes: 30
# RG2 (PREREGISTRATION_RG2.md): NG_D_s0 on rg2_test, gate none.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/rg2
[ -f results/rg2/NG_D_s0__rg2_test.summary.json ] && { echo done already; exit 0; }
rm -f results/rg2/NG_D_s0__rg2_test.jsonl
.venv/bin/python scripts/rg2_eval.py --model runs/p1_dwarm/merged --adapter runs/p1_d_s0/final --tag NG_D_s0 --gate none --split rg2_test > logs/rg2_NG_D_s0.log 2>&1 || { tail -40 logs/rg2_NG_D_s0.log; exit 1; }
tail -1 logs/rg2_NG_D_s0.log
sha256sum results/rg2/NG_D_s0__rg2_test.jsonl
echo RG2_NG_D_s0_OK
