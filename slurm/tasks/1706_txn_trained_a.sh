# minutes: 200
# TXN1 (PREREGISTRATION_TXN1.md): locked txn_test evaluation, trained_a.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_txn1.py --check || exit 1
mkdir -p logs results/txn
ev() { tag=$1; shift; [ -f results/txn/${tag}__txn_test.summary.json ] && { echo "skip $tag"; return 0; }; rm -f results/txn/${tag}__txn_test.jsonl; .venv/bin/python scripts/txn_eval.py --tag $tag --split txn_test --samples 2 "$@" > logs/txn_${tag}.log 2>&1 || { tail -30 logs/txn_${tag}.log; exit 1; }; tail -1 logs/txn_${tag}.log; }
ev W --model runs/rft_proc/merged
ev A_s0 --model runs/rft_proc/merged --adapter runs/s3_rv_mix_s0/final
ev A_s1 --model runs/rft_proc/merged --adapter runs/s3_rv_mix_s1/final
ev A_s2 --model runs/rft_proc/merged --adapter runs/s3_rv_mix_s2/final
ev Dw --model runs/p1_dwarm/merged
echo TXN_trained_a_OK
