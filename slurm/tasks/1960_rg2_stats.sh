# minutes: 15
# RG2 registered analysis (scripts/rg2_stats.py), once, after every RG2 run. CPU.
# after: 1950_rg2_V1_D_s0 1951_rg2_V2_D_s0 1952_rg2_V1_D_s1 1953_rg2_V2_D_s1 1954_rg2_V1_D_s2 1955_rg2_V2_D_s2 1956_rg2_NG_D_s0 1957_rg2_V2_SEL_s0 1958_rg2_NG_SEL_s0 1959_rg2_policies
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_rg2.py --check || exit 1
.venv/bin/python scripts/rg2_stats.py > logs/rg2_stats.log 2>&1 || { tail -20 logs/rg2_stats.log; exit 1; }
cat logs/rg2_stats.log
sha256sum results/rg2/rg2_summary.json
echo RG2_STATS_OK
