# minutes: 40
# RG1 (PREREGISTRATION_RG1.md): NG_aya8b on rg1_test, gate rules: none.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
bash slurm/rg1_run.sh NG_aya8b rg1_test none CohereLabs/aya-expanse-8b  || exit 1
echo RG1_NG_aya8b__rg1_test_OK
