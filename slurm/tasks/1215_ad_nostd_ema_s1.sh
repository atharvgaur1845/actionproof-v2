# minutes: 300
# after: 1213_ad_awrbn_ema_s0
# Algorithm discovery AD-D2 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL_D2.md): ad_nostd_ema_s1.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29215
source slurm/ad_arm_d2.sh nostd_ema 1 200
