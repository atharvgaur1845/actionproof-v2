# minutes: 30
# RG1 (PREREGISTRATION_RG1.md): RG_D_s1 on txn_test, gate rules: full.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
bash slurm/rg1_run.sh RG_D_s1 txn_test full runs/p1_dwarm/merged runs/p1_d_s1/final || exit 1
echo RG1_RG_D_s1__txn_test_OK
