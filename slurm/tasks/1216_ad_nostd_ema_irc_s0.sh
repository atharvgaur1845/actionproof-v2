# minutes: 300
# after: 1214_ad_awrbn_ema_s1
# Algorithm discovery AD-D3 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL_D3.md): ad_nostd_ema_irc_s0.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29216
source slurm/ad_arm_d3.sh nostd_ema_irc 0 200 smoke
