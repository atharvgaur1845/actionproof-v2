# minutes: 40
# Post-hoc validity analysis (exploratory; scripts/audit/txn_validity.py): reference policies FF/BD/BDE, same-key
# resolution of unresolvable tasks, net-money re-scoring of every TXN, SEL1 and P1 episode, DMR by origin. CPU only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/audit/txn_validity.py > logs/txn_validity.log 2>&1 || { tail -30 logs/txn_validity.log; exit 1; }
cat results/audit/txn_validity.md
sha256sum results/audit/txn_validity.json results/audit/txn_validity.md
echo TXN_VALIDITY_OK
