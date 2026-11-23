# minutes: 300
# after: 1211_ad_awrbnc30_s0
# Algorithm discovery AD-D2 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL_D2.md): ad_awrbn_ema_s0.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29213
source slurm/ad_arm_d2.sh awrbn_ema 0 200
