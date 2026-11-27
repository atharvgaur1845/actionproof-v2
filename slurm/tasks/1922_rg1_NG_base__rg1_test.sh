# minutes: 30
# RG1 (PREREGISTRATION_RG1.md): NG_base on rg1_test, gate rules: none.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
bash slurm/rg1_run.sh NG_base rg1_test none CohereLabs/tiny-aya-global  || exit 1
echo RG1_NG_base__rg1_test_OK
