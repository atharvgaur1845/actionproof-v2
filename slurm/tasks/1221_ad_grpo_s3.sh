# minutes: 300
# after: 1217_ad_nostd_ema_irc_s1
# AD-C1 (PREREGISTRATION_AD_C1.md): ad_grpo_s3, training only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29221
source slurm/ad_arm_c1.sh grpo 3
