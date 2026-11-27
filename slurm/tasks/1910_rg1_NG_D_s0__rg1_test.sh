# minutes: 30
# RG1 (PREREGISTRATION_RG1.md): NG_D_s0 on rg1_test, gate rules: none.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
bash slurm/rg1_run.sh NG_D_s0 rg1_test none runs/p1_dwarm/merged runs/p1_d_s0/final || exit 1
echo RG1_NG_D_s0__rg1_test_OK
