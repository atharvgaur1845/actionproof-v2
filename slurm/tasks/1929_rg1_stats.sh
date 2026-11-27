# minutes: 30
# RG1 registered analysis (scripts/rg1_stats.py), once the confirmatory arms and the scripted arms exist. CPU.
# after: 1910_rg1_NG_D_s0__rg1_test 1911_rg1_RG_D_s0__rg1_test 1912_rg1_NG_D_s1__rg1_test 1913_rg1_RG_D_s1__rg1_test 1914_rg1_NG_D_s2__rg1_test 1915_rg1_RG_D_s2__rg1_test 1916_rg1_policies
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_rg1.py --check || exit 1
.venv/bin/python scripts/rg1_stats.py > logs/rg1_stats.log 2>&1 || { tail -30 logs/rg1_stats.log; exit 1; }
cat logs/rg1_stats.log
sha256sum results/rg1/rg1_summary.json results/rg1/rg1_summary.md
echo RG1_STATS_OK
