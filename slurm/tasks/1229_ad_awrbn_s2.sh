# minutes: 300
# after: 1227_ad_nostd_s3
# AD-C1 (PREREGISTRATION_AD_C1.md): ad_awrbn_s2, training only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29229
source slurm/ad_arm_c1.sh awrbn 2
