# minutes: 30
# RG2 (PREREGISTRATION_RG2.md): V2_D_s1 on rg2_test, gate v2.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/rg2
[ -f results/rg2/V2_D_s1__rg2_test.summary.json ] && { echo done already; exit 0; }
rm -f results/rg2/V2_D_s1__rg2_test.jsonl
.venv/bin/python scripts/rg2_eval.py --model runs/p1_dwarm/merged --adapter runs/p1_d_s1/final --tag V2_D_s1 --gate v2 --split rg2_test > logs/rg2_V2_D_s1.log 2>&1 || { tail -40 logs/rg2_V2_D_s1.log; exit 1; }
tail -1 logs/rg2_V2_D_s1.log
sha256sum results/rg2/V2_D_s1__rg2_test.jsonl
echo RG2_V2_D_s1_OK
