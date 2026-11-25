# minutes: 200
# TXN1 (PREREGISTRATION_TXN1.md): locked txn_test evaluation, trained_d.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_txn1.py --check || exit 1
mkdir -p logs results/txn
ev() { tag=$1; shift; [ -f results/txn/${tag}__txn_test.summary.json ] && { echo "skip $tag"; return 0; }; rm -f results/txn/${tag}__txn_test.jsonl; .venv/bin/python scripts/txn_eval.py --tag $tag --split txn_test --samples 2 "$@" > logs/txn_${tag}.log 2>&1 || { tail -30 logs/txn_${tag}.log; exit 1; }; tail -1 logs/txn_${tag}.log; }
ev D_s0 --model runs/p1_dwarm/merged --adapter runs/p1_d_s0/final
ev D_s1 --model runs/p1_dwarm/merged --adapter runs/p1_d_s1/final
ev D_s2 --model runs/p1_dwarm/merged --adapter runs/p1_d_s2/final
ev Dn_s0 --model runs/p1_dwarm/merged --adapter runs/p3_dn_s0/final
ev Dn_s1 --model runs/p1_dwarm/merged --adapter runs/p3_dn_s1/final
ev Dn_s2 --model runs/p1_dwarm/merged --adapter runs/p3_dn_s2/final
echo TXN_trained_d_OK
