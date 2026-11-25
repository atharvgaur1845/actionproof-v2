# minutes: 15
# after: 1720_p5_fire 1721_p5_water 1722_p5_earth 1723_p5_expanse8b 1701_b1_aya8b
# P5 registered analysis, run once.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p5.py --check || exit 1
[ -f results/p5/p5_summary.json ] && { echo "already analysed"; exit 0; }
.venv/bin/python scripts/p5_stats.py || exit 1
echo P5_STATS_OK
