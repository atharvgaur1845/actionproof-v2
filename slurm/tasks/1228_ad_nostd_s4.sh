# minutes: 300
# after: 1226_ad_nostd_s2
# AD-C1 (PREREGISTRATION_AD_C1.md): ad_nostd_s4, training only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29228
source slurm/ad_arm_c1.sh nostd 4
