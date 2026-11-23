# minutes: 300
# after: 1228_ad_nostd_s4
# AD-C1 (PREREGISTRATION_AD_C1.md): ad_awrbn_s3, training only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29230
source slurm/ad_arm_c1.sh awrbn 3
