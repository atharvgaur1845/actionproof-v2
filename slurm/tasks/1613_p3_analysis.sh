# minutes: 20
# after: 1610_p3_dn_s0 1611_p3_dn_s1 1612_p3_dn_s2
# P3 registered analysis, run once.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p3.py --check || exit 1
.venv/bin/python scripts/p3_stats.py > logs/p3_stats.log 2>&1 || { tail -20 logs/p3_stats.log; exit 1; }
cat logs/p3_stats.log
