# minutes: 40
# RG1 (PREREGISTRATION_RG1.md): NG_SEL_s0 on rg1_test, gate rules: none.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
bash slurm/rg1_run.sh NG_SEL_s0 rg1_test none runs/p4_dwaug_s0/merged runs/sel1_SEL_s0/final || exit 1
echo RG1_NG_SEL_s0__rg1_test_OK
