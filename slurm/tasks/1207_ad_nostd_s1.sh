# minutes: 300
# after: 1205_ad_awrb_s1
# Algorithm discovery AD-D1 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL.md): ad_nostd_s1.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29207
source slurm/ad_arm.sh nostd 1 200
