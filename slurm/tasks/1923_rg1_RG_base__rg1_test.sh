# minutes: 30
# RG1 (PREREGISTRATION_RG1.md): RG_base on rg1_test, gate rules: full.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
bash slurm/rg1_run.sh RG_base rg1_test full CohereLabs/tiny-aya-global  || exit 1
echo RG1_RG_base__rg1_test_OK
