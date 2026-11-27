# minutes: 30
# RG1 registered analysis (scripts/rg1_stats.py), run once after every registered RG1 run has finished. CPU.
# after: 1910_rg1_NG_D_s0__rg1_test 1911_rg1_RG_D_s0__rg1_test 1912_rg1_NG_D_s1__rg1_test 1913_rg1_RG_D_s1__rg1_test 1914_rg1_NG_D_s2__rg1_test 1915_rg1_RG_D_s2__rg1_test 1916_rg1_policies 1917_rg1_AB_no_verify_D_s0__rg1_test 1918_rg1_AB_no_retry_D_s0__rg1_test 1919_rg1_AB_no_fresh_D_s0__rg1_test 1920_rg1_AB_no_repair_D_s0__rg1_test 1921_rg1_AB_no_escalate_D_s0__rg1_test 1922_rg1_NG_base__rg1_test 1923_rg1_RG_base__rg1_test 1924_rg1_NG_Dwaug_s0__rg1_test 1925_rg1_RG_Dwaug_s0__rg1_test 1926_rg1_RG_D_s0__txn_test 1927_rg1_RG_D_s1__txn_test 1928_rg1_RG_D_s2__txn_test 1930_rg1_NG_SEL_s0__rg1_test 1931_rg1_RG_SEL_s0__rg1_test 1932_rg1_NG_aya8b__rg1_test 1933_rg1_RG_aya8b__rg1_test
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_rg1.py --check || exit 1
.venv/bin/python scripts/rg1_stats.py > logs/rg1_stats.log 2>&1 || { tail -30 logs/rg1_stats.log; exit 1; }
cat logs/rg1_stats.log
sha256sum results/rg1/rg1_summary.json results/rg1/rg1_summary.md
echo RG1_STATS_OK
