# minutes: 30
# Descriptive (not registered): TXN metrics for every tag including Dw_aug s0-s2 (P4 models, evaluated after 1739).
# Writes a separate file; the registered results/txn/stats__txn_test.{json,md} from 1739 is not touched.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_txn1.py --check || exit 1
.venv/bin/python scripts/txn_stats.py --split txn_test --out results/txn/stats_all_descriptive__txn_test.json \
  --pairs Dwaug_s0:Dw,Dwaug_s1:Dw,Dwaug_s2:Dw || exit 1
echo TXN_STATS_ALL_OK
