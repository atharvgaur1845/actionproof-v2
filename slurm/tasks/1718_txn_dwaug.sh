# minutes: 200
# TXN1 (PREREGISTRATION_TXN1.md): locked txn_test evaluation, dwaug.
# after: 1711_p4_Dwaug_s0 1712_p4_Dwaug_s1 1713_p4_Dwaug_s2
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_txn1.py --check || exit 1
mkdir -p logs results/txn
ev() { tag=$1; shift; [ -f results/txn/${tag}__txn_test.summary.json ] && { echo "skip $tag"; return 0; }; rm -f results/txn/${tag}__txn_test.jsonl; .venv/bin/python scripts/txn_eval.py --tag $tag --split txn_test --samples 2 "$@" > logs/txn_${tag}.log 2>&1 || { tail -30 logs/txn_${tag}.log; exit 1; }; tail -1 logs/txn_${tag}.log; }
ev Dwaug_s0 --model runs/p4_dwaug_s0/merged
ev Dwaug_s1 --model runs/p4_dwaug_s1/merged
ev Dwaug_s2 --model runs/p4_dwaug_s2/merged
echo TXN_dwaug_OK
