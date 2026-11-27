# minutes: 30
# RG1 (PREREGISTRATION_RG1.md): AB_no_verify_D_s0 on rg1_test, gate rules: no_verify.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
bash slurm/rg1_run.sh AB_no_verify_D_s0 rg1_test no_verify runs/p1_dwarm/merged runs/p1_d_s0/final || exit 1
echo RG1_AB_no_verify_D_s0__rg1_test_OK
