# minutes: 300
# after: 1223_ad_awrb_s2
# AD-C1 (PREREGISTRATION_AD_C1.md): ad_awrb_s4, training only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29225
source slurm/ad_arm_c1.sh awrb 4
