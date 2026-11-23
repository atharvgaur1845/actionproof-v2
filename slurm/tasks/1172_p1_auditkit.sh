# minutes: 90
# P1 secondary (PREREGISTRATION_P1.md §6, D-P1-1): AuditKIT TaskCompletion (Aya Expanse 32B) on seed-0 greedy test episodes of every arm; descriptive only
# after: 1160_p1_test_A_s0 1161_p1_test_B_s0 1162_p1_test_C_s0 1163_p1_test_D_s0
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export PYTHONPATH=vendor/AuditKIT/src:${PYTHONPATH:-}
.venv/bin/python scripts/p1_auditkit.py > logs/p1_auditkit.full.log 2>&1 || { tail -40 logs/p1_auditkit.full.log; exit 1; }
tail -30 logs/p1_auditkit.full.log
