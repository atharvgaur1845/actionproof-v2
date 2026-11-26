# minutes: 30
# Descriptive (not registered): rerun of 1724 adding qwen (1725), plus a compact extract for the docs.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_txn1.py --check || exit 1
.venv/bin/python scripts/txn_stats.py --split txn_test --out results/txn/stats_all_descriptive__txn_test.json \
  --pairs Dwaug_s0:Dw,Dwaug_s1:Dw,Dwaug_s2:Dw || exit 1
.venv/bin/python scripts/txn_compact.py results/txn/stats_all_descriptive__txn_test.json results/txn/compact_descriptive__txn_test.json || exit 1
echo TXN_STATS_ALL2_OK
