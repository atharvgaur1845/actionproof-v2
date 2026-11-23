# minutes: 300
# after: 1207_ad_nostd_s1
# Algorithm discovery AD-D1 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL.md): ad_awrbc30_s0.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29209
source slurm/ad_arm.sh awrb 0 200 0.30 c30
