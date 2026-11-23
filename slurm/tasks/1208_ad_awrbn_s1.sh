# minutes: 300
# after: 1206_ad_grpo_s1
# Algorithm discovery AD-D1 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL.md): ad_awrbn_s1.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29208
source slurm/ad_arm.sh awrbn 1 200
