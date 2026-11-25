# minutes: 15
# after: 1711_p4_Dwaug_s0 1712_p4_Dwaug_s1 1713_p4_Dwaug_s2 1714_p4_Dw_s1 1715_p4_Dw_s2 1716_p4_Dw_s0
# P4 registered analysis, run once (scripts/p4_stats.py).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p4.py --check || exit 1
[ -f results/p4/p4_summary.json ] && { echo "already analysed"; exit 0; }
.venv/bin/python scripts/p4_stats.py || exit 1
echo P4_STATS_OK
