# minutes: 20
# TXN1: scripted policies B0-B4, A1-A6 on the locked txn_test (CPU, deterministic).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_txn1.py --check || exit 1
.venv/bin/python -m pytest -q tests/txn > logs/txn_tests.log 2>&1 || { tail -20 logs/txn_tests.log; exit 1; }
tail -1 logs/txn_tests.log
.venv/bin/python scripts/txn_make_splits.py --check | tail -2
[ -f results/txn/B4__txn_test.jsonl ] || .venv/bin/python scripts/txn_policies_eval.py --split txn_test --table || exit 1
echo TXN_POLICIES_OK
