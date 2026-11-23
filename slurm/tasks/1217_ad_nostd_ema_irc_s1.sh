# minutes: 300
# after: 1215_ad_nostd_ema_s1
# Algorithm discovery AD-D3 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL_D3.md): ad_nostd_ema_irc_s1.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29217
source slurm/ad_arm_d3.sh nostd_ema_irc 1 200
