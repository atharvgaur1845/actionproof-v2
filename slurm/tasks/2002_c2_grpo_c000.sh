# minutes: 150
# AD-C2 (PREREGISTRATION_AD_C2.md): GRPO at read cost 0 (boundary arm), seed 0, training only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29302
source slurm/ad_arm_c2.sh grpo c000 0.0 0
