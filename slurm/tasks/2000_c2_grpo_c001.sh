# minutes: 150
# AD-C2 (PREREGISTRATION_AD_C2.md): GRPO at read cost 0.01, seed 0, training only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29300
source slurm/ad_arm_c2.sh grpo c001 0.01 0
