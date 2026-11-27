# minutes: 90
# SEL1 (PREREGISTRATION_SEL1.md), descriptive: the parents Dw_aug s0-s2 and TXN1's D seed 0 on the sealed sel1_test.
# after: 1800_sel1_build
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_sel1.py --check || exit 1
mkdir -p logs results/sel1
ev() { tag=$1; shift; [ -f results/sel1/${tag}__sel1_test.summary.json ] && { echo "skip $tag"; return 0; }; rm -f results/sel1/${tag}__sel1_test.jsonl; .venv/bin/python scripts/sel1_eval.py --tag $tag --split sel1_test --samples 2 "$@" > logs/sel1_${tag}.log 2>&1 || { tail -30 logs/sel1_${tag}.log; exit 1; }; tail -1 logs/sel1_${tag}.log; }
ev Dwaug_s0 --model runs/p4_dwaug_s0/merged
ev Dwaug_s1 --model runs/p4_dwaug_s1/merged
ev Dwaug_s2 --model runs/p4_dwaug_s2/merged
ev D_s0 --model runs/p1_dwarm/merged --adapter runs/p1_d_s0/final
echo SEL1_parents_OK
