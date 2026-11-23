# minutes: 300
# after: 1208_ad_awrbn_s1
# Algorithm discovery AD-D1 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL.md): ad_awrbnc30_s0.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29211
source slurm/ad_arm.sh awrbn 0 200 0.30 c30
