# minutes: 300
# after: 1225_ad_awrb_s4
# AD-C1 (PREREGISTRATION_AD_C1.md): ad_nostd_s3, training only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29227
source slurm/ad_arm_c1.sh nostd 3
