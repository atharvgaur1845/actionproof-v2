# minutes: 30
# RG1 (PREREGISTRATION_RG1.md): RG_Dwaug_s0 on rg1_test, gate rules: full.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
bash slurm/rg1_run.sh RG_Dwaug_s0 rg1_test full runs/p4_dwaug_s0/merged  || exit 1
echo RG1_RG_Dwaug_s0__rg1_test_OK
