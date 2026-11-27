# minutes: 20
# SEL1 (PREREGISTRATION_SEL1.md): the registered analysis (scripts/sel1_stats.py), once every arm has finished.
# after: 1801_sel1_SEL_s0 1802_sel1_CTL_s0 1803_sel1_SEL_s1 1804_sel1_CTL_s1 1805_sel1_SEL_s2 1806_sel1_CTL_s2 1807_sel1_parents
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_sel1.py --check || exit 1
.venv/bin/python scripts/sel1_stats.py > logs/sel1_stats.log 2>&1 || { tail -30 logs/sel1_stats.log; exit 1; }
cat logs/sel1_stats.log
cat results/sel1/sel1_summary.md
echo SEL1_STATS_OK
