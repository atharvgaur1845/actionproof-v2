# minutes: 20
# after: 1704_txn_policies 1706_txn_trained_a 1707_txn_trained_d 1708_txn_prompted
# TXN1 registered analysis (scripts/txn_stats.py). Rerun later to add aya32b/Dwaug/qwen descriptively; H3 uses only D and B1/B3.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_txn1.py --check || exit 1
.venv/bin/python scripts/txn_stats.py --split txn_test || exit 1
.venv/bin/python scripts/txn_invariants_recheck.py --split txn_test || exit 1  # D-TXN1-1, addendum 1
echo TXN_STATS_OK
