# minutes: 30
# RG2 (PREREGISTRATION_RG2.md): V2_SEL_s0 on rg2_test, gate v2.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/rg2
[ -f results/rg2/V2_SEL_s0__rg2_test.summary.json ] && { echo done already; exit 0; }
rm -f results/rg2/V2_SEL_s0__rg2_test.jsonl
.venv/bin/python scripts/rg2_eval.py --model runs/p4_dwaug_s0/merged --adapter runs/sel1_SEL_s0/final --tag V2_SEL_s0 --gate v2 --split rg2_test > logs/rg2_V2_SEL_s0.log 2>&1 || { tail -40 logs/rg2_V2_SEL_s0.log; exit 1; }
tail -1 logs/rg2_V2_SEL_s0.log
sha256sum results/rg2/V2_SEL_s0__rg2_test.jsonl
echo RG2_V2_SEL_s0_OK
