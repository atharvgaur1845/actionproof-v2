# minutes: 300
# after: 1214_ad_awrbn_ema_s1
# AD-C1 (PREREGISTRATION_AD_C1.md): ad_grpo_s2, training only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29220
source slurm/ad_arm_c1.sh grpo 2
