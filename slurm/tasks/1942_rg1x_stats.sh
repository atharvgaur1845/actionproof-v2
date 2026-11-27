# minutes: 15
# RG1 addendum 1 descriptive table (scripts/rg1x_stats.py). CPU.
# after: 1940_rg1x_X1_RG_D_s0 1941_rg1x_X1_RG_SEL_s0
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/rg1x_stats.py > logs/rg1x_stats.log 2>&1 || { tail -20 logs/rg1x_stats.log; exit 1; }
cat logs/rg1x_stats.log
echo RG1X_STATS_OK
