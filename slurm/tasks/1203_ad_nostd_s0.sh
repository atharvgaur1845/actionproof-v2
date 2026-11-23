# minutes: 300
# after: 1201_ad_awrb_s0
# Algorithm discovery AD-D1 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL.md): ad_nostd_s0.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29203
source slurm/ad_arm.sh nostd 0 200
